;;; nl-llm-agent-recur-supervised.el --- completion supervision for recurrent models -*- lexical-binding: t; -*-

;;; Commentary:
;; A small CPU-only bridge from the public completion example format to the
;; recurrent-depth model.  It deliberately does not create models, enable a
;; backend, checkpoint, or contact a provider.

;;; Code:

(require 'cl-lib)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-recur)
(require 'nl-llm-agent-supervised)
(require 'nl-llm-agent-tokenizer)

(defconst nl-llm-agent-recur-supervised--max-seed #xffffffff)
(defconst nl-llm-agent-recur-supervised--allowed-loss-keys
  '(:r :k :s0-seed :tokenizer))
(defconst nl-llm-agent-recur-supervised--allowed-train-keys
  '(:r :k :s0-seed :tokenizer :lr :epochs))
(defconst nl-llm-agent-recur-supervised--allowed-model-keys
  '(:wte :prelude :wa :ba :core :coda :lnfg :wh :bh :dim :heads
    :kv-heads :sigma :vocab :tokenizer :ff))

(defun nl-llm-agent-recur-supervised--gpu-active-p ()
  "Return non-nil when any known photon tensor op is GPU-swapped."
  (let ((pairs (if (boundp 'photon-tensor-gpu--ops)
                   photon-tensor-gpu--ops
                 '((photon-tensor-matmul . photon-tensor-matmul-gpu)
                   (photon-tensor-linear . photon-tensor-linear-gpu)
                   (photon-tensor-softmax-rows . photon-tensor-softmax-rows-gpu)
                   (photon-tensor-layernorm-rows . photon-tensor-layernorm-rows-gpu)
                   (photon-tensor-gelu . photon-tensor-gelu-gpu)))))
    (catch 'active
      (dolist (pair pairs)
        (when (and (fboundp (car pair)) (fboundp (cdr pair))
                   (eq (indirect-function (car pair))
                       (indirect-function (cdr pair))))
          (throw 'active t)))
      nil)))

(defun nl-llm-agent-recur-supervised--proper-list-p (value)
  "Return non-nil when VALUE is a finite proper list, detecting cycles."
  (let ((slow value) (fast value) (ok t))
    (while (and ok (consp fast))
      (setq fast (cdr fast))
      (when (consp fast)
        (setq fast (cdr fast) slow (cdr slow))
        (when (eq fast slow) (setq ok nil))))
    (and ok (null fast))))

(defun nl-llm-agent-recur-supervised--finite-p (value)
  "Return non-nil for a finite real NUMBER."
  (and (numberp value)
       (condition-case nil
           (and (= value value) (< (abs (float value)) 1.0e308))
         (error nil))))

(defun nl-llm-agent-recur-supervised--keys (keys allowed where)
  "Validate exact plist KEYS against ALLOWED for WHERE."
  (unless (nl-llm-agent-recur-supervised--proper-list-p keys)
    (error "%s must be a proper plist" where))
  (let ((tail keys) seen)
    (while tail
      (let ((key (pop tail)))
        (unless (memq key allowed)
          (error "%s has unknown key %S" where key))
        (when (null tail)
          (error "%s has an unpaired key %S" where key))
        (when (memq key seen)
          (error "%s has duplicate key %S" where key))
        (push key seen)
        (pop tail))))
  keys)

(defun nl-llm-agent-recur-supervised--validate-examples (examples)
  "Reject cyclic or malformed example containers before supervised prepare."
  (unless (and (vectorp examples) (> (length examples) 0)
               (<= (length examples)
                   nl-llm-agent-supervised-max-examples))
    (error "recurrent supervised examples must be a bounded vector"))
  (dotimes (index (length examples))
    (unless (nl-llm-agent-recur-supervised--proper-list-p
             (aref examples index))
      (error "recurrent supervised example %d must be a proper plist" index)))
  examples)

(defun nl-llm-agent-recur-supervised--tensor-p
    (value shape where)
  "Validate PAV VALUE has exact tensor SHAPE and finite data for WHERE."
  (unless (pav-p value)
    (error "%s must be a PAV" where))
  (let* ((tensor (pav-value value))
         (actual-shape (and (vectorp tensor) (= (length tensor) 2)
                            (aref tensor 0)))
         (data (and (vectorp tensor) (= (length tensor) 2)
                    (aref tensor 1)))
         (size (and (nl-llm-agent-recur-supervised--proper-list-p shape)
                    (let ((product 1) (tail shape))
                      (while tail
                        (let ((dimension (pop tail)))
                          (unless (and (integerp dimension) (> dimension 0))
                            (error "%s has invalid expected shape" where))
                          (setq product (* product dimension))))
                      product))))
    (unless (and (vectorp tensor) (= (length tensor) 2)
                 (nl-llm-agent-recur-supervised--proper-list-p actual-shape)
                 (equal actual-shape shape)
                 (vectorp data) (= (length data) size))
      (error "%s has invalid tensor shape or data length" where))
    (let ((index 0))
      (while (< index (length data))
        (unless (nl-llm-agent-recur-supervised--finite-p (aref data index))
          (error "%s contains a non-finite value" where))
        (setq index (1+ index))))
    t))

(defun nl-llm-agent-recur-supervised--block
    (block dim ff heads kv-heads where)
  "Validate one recurrent BLOCK and its parameter shapes."
  (unless (nl-llm-agent-recur-supervised--proper-list-p block)
    (error "%s must be a proper plist" where))
  (nl-llm-agent-recur-supervised--keys
   block nl-llm-recur--block-keys where)
  (let ((kvdim (* kv-heads (/ dim heads))))
    (dolist (key nl-llm-recur--block-keys)
      (let ((shape
             (pcase key
               ((or :ln1g :ln2g) (list dim))
               (:wq (list dim dim)) (:bq (list dim))
               (:wk (list kvdim dim)) (:bk (list kvdim))
               (:wv (list kvdim dim)) (:bv (list kvdim))
               (:wo (list dim dim)) (:bo (list dim))
               (:wg (list ff dim)) (:bg (list ff))
               (:wu (list ff dim)) (:bu (list ff))
               (:wd (list dim ff)) (:bd (list dim)))))
        (nl-llm-agent-recur-supervised--tensor-p
         (plist-get block key) shape (format "%s %S" where key)))))
  t)

(defun nl-llm-agent-recur-supervised--model
    (model tokenizer)
  "Validate MODEL for TOKENIZER and return geometry metadata."
  (unless (nl-llm-agent-recur-supervised--proper-list-p model)
    (error "recurrent supervised model must be a proper plist"))
  (nl-llm-agent-recur-supervised--keys
   model nl-llm-agent-recur-supervised--allowed-model-keys
   "recurrent supervised model")
  (let* ((dim (plist-get model :dim))
         (heads (plist-get model :heads))
         (kv-heads (plist-get model :kv-heads))
         (sigma (plist-get model :sigma))
         (wte (plist-get model :wte)))
    (unless (and (integerp dim) (> dim 0)
                 (integerp heads) (> heads 0)
                 (= (% dim heads) 0)
                 (cl-evenp (/ dim heads))
                 (integerp kv-heads) (> kv-heads 0)
                 (= (% heads kv-heads) 0))
      (error "recurrent supervised model has invalid head geometry"))
    (unless (and (nl-llm-agent-recur-supervised--finite-p sigma)
                 (>= sigma 0.0))
      (error "recurrent supervised model sigma must be finite and nonnegative"))
    (let* ((wte-tensor (and (pav-p wte) (pav-value wte)))
           (wte-shape (and (vectorp wte-tensor) (= (length wte-tensor) 2)
                           (aref wte-tensor 0)))
           (vocab (and (nl-llm-agent-recur-supervised--proper-list-p wte-shape)
                       (car wte-shape))))
      (unless (and (integerp vocab) (> vocab 0))
        (error "recurrent supervised model has no valid vocabulary tensor"))
      (nl-llm-agent-recur-supervised--tensor-p wte (list vocab dim) "wte")
      (when (plist-member model :vocab)
        (unless (and (integerp (plist-get model :vocab))
                     (= (plist-get model :vocab) vocab))
          (error "recurrent supervised model vocab metadata disagrees")))
      (when (plist-member model :tokenizer)
        (unless (equal (nl-llm-agent-tokenizer-id (plist-get model :tokenizer))
                       tokenizer)
          (error "recurrent supervised model tokenizer disagrees")))
      (unless (= vocab (nl-llm-agent-tokenizer-vocab tokenizer))
        (error "recurrent supervised model vocab disagrees with tokenizer"))
      (let* ((ff nil)
             (core (plist-get model :core))
             (prelude (plist-get model :prelude))
             (coda (plist-get model :coda)))
        (dolist (stack (list prelude core coda))
          (unless (nl-llm-agent-recur-supervised--proper-list-p stack)
            (error "recurrent supervised block stack must be a proper list")))
        (unless (consp core)
          (error "recurrent supervised model core must be nonempty"))
        (dolist (stack (list prelude core coda))
          (dolist (block stack)
            (unless (nl-llm-agent-recur-supervised--proper-list-p block)
              (error "recurrent supervised block must be a proper plist"))
            (let ((block-wg (plist-get block :wg)))
              (when (null ff)
                (let ((shape (and (pav-p block-wg)
                                  (photon-tensor-shape (pav-value block-wg)))))
                  (setq ff
                        (and (nl-llm-agent-recur-supervised--proper-list-p shape)
                             (equal (length shape) 2)
                             (car shape)))))
              (unless (and (integerp ff) (> ff 0))
                (error "recurrent supervised model has invalid feedforward shape"))
              (nl-llm-agent-recur-supervised--block
               block dim ff heads kv-heads "recurrent block"))))
        (nl-llm-agent-recur-supervised--tensor-p
         (plist-get model :wa) (list dim (* 2 dim)) "adapter weight")
        (nl-llm-agent-recur-supervised--tensor-p
         (plist-get model :ba) (list dim) "adapter bias")
        (nl-llm-agent-recur-supervised--tensor-p
         (plist-get model :lnfg) (list dim) "final norm")
        (nl-llm-agent-recur-supervised--tensor-p
         (plist-get model :wh) (list vocab dim) "head weight")
        (nl-llm-agent-recur-supervised--tensor-p
         (plist-get model :bh) (list vocab) "head bias")
        (when (plist-member model :ff)
          (unless (and (integerp (plist-get model :ff))
                       (= (plist-get model :ff) ff))
            (error "recurrent supervised model ff metadata disagrees")))
        (list :dim dim :heads heads :kv-heads kv-heads :sigma sigma
              :vocab vocab :ff ff)))))

(defun nl-llm-agent-recur-supervised--parse-options (keys trainp)
  "Validate and normalize KEYS for LOSS or TRAIN according to TRAINP."
  (nl-llm-agent-recur-supervised--keys
   keys (if trainp
            nl-llm-agent-recur-supervised--allowed-train-keys
          nl-llm-agent-recur-supervised--allowed-loss-keys)
   "recurrent supervised options")
  (let* ((r (if (plist-member keys :r) (plist-get keys :r) 2))
         (k (if (plist-member keys :k) (plist-get keys :k) r))
         (seed (if (plist-member keys :s0-seed)
                   (plist-get keys :s0-seed) 1))
         (tokenizer-value
          (if (plist-member keys :tokenizer)
              (plist-get keys :tokenizer)
            nl-llm-agent-tokenizer-utf8))
         (tokenizer
          (and (stringp tokenizer-value)
               (nl-llm-agent-tokenizer-id tokenizer-value)))
         (lr (and trainp (if (plist-member keys :lr)
                             (plist-get keys :lr) 0.01)))
         (epochs (and trainp (if (plist-member keys :epochs)
                                 (plist-get keys :epochs) 1))))
    (unless (and (integerp r) (<= 1 r) (<= r 32))
      (error "recurrent supervised :r must be an integer in 1..32"))
    (unless (and (integerp k) (<= 1 k) (<= k r))
      (error "recurrent supervised :k must be an integer in 1..r"))
    (unless (and (integerp seed) (<= 0 seed)
                 (<= seed nl-llm-agent-recur-supervised--max-seed))
      (error "recurrent supervised :s0-seed must be a uint32"))
    (unless tokenizer
      (error "recurrent supervised :tokenizer must be a supported string"))
    (when trainp
      (unless (and (nl-llm-agent-recur-supervised--finite-p lr)
                   (> lr 0.0) (<= lr 1.0))
        (error "recurrent supervised :lr must be finite and in (0,1]"))
      (unless (and (integerp epochs) (<= 1 epochs) (<= epochs 32))
        (error "recurrent supervised :epochs must be an integer in 1..32")))
    (list :r r :k k :s0-seed seed :tokenizer tokenizer
          :lr lr :epochs epochs)))

(defun nl-llm-agent-recur-supervised--s0 (geometry tokens seed)
  "Build a deterministic fresh S0 for TOKENS from SEED."
  (photon-autograd-const
   (photon-tensor
    (list (length tokens) (plist-get geometry :dim))
    (nl-llm-recur-randn
     (* (length tokens) (plist-get geometry :dim))
     (plist-get geometry :sigma) seed))))

(defun nl-llm-agent-recur-supervised--loss-one
    (model geometry trajectory start options)
  "Return one masked completion loss for TRAJECTORY."
  (let* ((tokens (butlast trajectory))
         (targets-mask
          (nl-llm-agent-supervised--targets-mask trajectory start))
         (forward
          (nl-llm-recur-forward
           model tokens (plist-get options :r) :k (plist-get options :k)
           :s0 (nl-llm-agent-recur-supervised--s0
                geometry tokens (plist-get options :s0-seed)))))
    (nl-llm-ag-masked-softmax-ce
     (plist-get forward :logits)
     (car targets-mask) (cdr targets-mask))))

(defun nl-llm-agent-recur-supervised--loss-plan
    (model geometry plan options)
  "Return completion-token-weighted loss for MODEL over PLAN."
  (let ((weighted 0.0)
        (index 0)
        (trajectories (plist-get plan :trajectories))
        (loss-starts (plist-get plan :loss-starts)))
    ;; A dynamic nil tape makes this read-only path preserve the caller's tape.
    (let ((photon-autograd--tape nil))
      (dolist (trajectory trajectories)
        (setq photon-autograd--tape nil)
        (let* ((start (aref loss-starts index))
               (loss
                (nl-llm-agent-recur-supervised--loss-one
                 model geometry trajectory start options))
               (value (aref (photon-tensor-data (pav-value loss)) 0))
               (active (- (length trajectory) start)))
          (unless (nl-llm-agent-recur-supervised--finite-p value)
            (error "recurrent supervised loss is non-finite"))
          (setq weighted (+ weighted (* active value))
                index (1+ index))))
      (/ weighted (float (plist-get plan :completion-tokens))))))

(defun nl-llm-agent-recur-supervised--all-finite-p (parameters)
  "Return non-nil when all PARAMETER values and gradients are finite."
  (catch 'bad
    (dolist (parameter parameters)
      (let ((value (pav-value parameter))
            (gradient (pav-grad parameter)))
        (dolist (tensor (list value gradient))
          (unless (and (vectorp tensor) (= (length tensor) 2)
                       (vectorp (aref tensor 1)))
            (throw 'bad nil))
          (let ((data (aref tensor 1)) (index 0))
            (while (< index (length data))
              (unless (nl-llm-agent-recur-supervised--finite-p
                       (aref data index))
                (throw 'bad nil))
              (setq index (1+ index)))))))
    t))

;;;###autoload
(cl-defun nl-llm-agent-recur-supervised-loss (model examples &rest keys)
  "Return completion-token-weighted CPU loss for recurrent MODEL.

KEYS are :r, :k, :s0-seed, and :tokenizer.  The deterministic S0 is local to
each forward.  This adapter never mutates MODEL's weights or gradients and
does not support providers, checkpoints, or GPU execution."
  (let* ((options (nl-llm-agent-recur-supervised--parse-options keys nil))
         (_gpu
          (when (nl-llm-agent-recur-supervised--gpu-active-p)
            (error "recurrent supervised CPU path refuses an active GPU")))
         (geometry
          (nl-llm-agent-recur-supervised--model
           model (plist-get options :tokenizer)))
         (plan
          (nl-llm-agent-supervised--prepare
           (nl-llm-agent-recur-supervised--validate-examples examples)
           (plist-get options :tokenizer))))
    (nl-llm-agent-recur-supervised--loss-plan
     model geometry plan options)))

;;;###autoload
(cl-defun nl-llm-agent-recur-supervised-train (model examples &rest keys)
  "Train recurrent MODEL on completion-only EXAMPLES with sequential CPU SGD.

KEYS accepts loss options plus :lr and :epochs.  Validation completes before
the first update.  The fixed-seed S0 bridge is deterministic and has no
checkpoint or provider support; a later numerical failure does not roll back
updates that already completed."
  (let* ((options (nl-llm-agent-recur-supervised--parse-options keys t))
         (_gpu
          (when (nl-llm-agent-recur-supervised--gpu-active-p)
            (error "recurrent supervised CPU path refuses an active GPU")))
         (geometry
          (nl-llm-agent-recur-supervised--model
           model (plist-get options :tokenizer)))
         (plan
          (nl-llm-agent-supervised--prepare
           (nl-llm-agent-recur-supervised--validate-examples examples)
           (plist-get options :tokenizer)))
         (parameters (nl-llm-recur-params model))
         (loss-before
          (nl-llm-agent-recur-supervised--loss-plan
           model geometry plan options)))
    (let ((photon-autograd--tape nil)
          (epoch 0))
      (while (< epoch (plist-get options :epochs))
        (let ((index 0))
          (dolist (trajectory (plist-get plan :trajectories))
            (let ((start (aref (plist-get plan :loss-starts) index)))
              (setq photon-autograd--tape nil)
              (photon-autograd-zero-grad parameters)
              (let ((loss
                     (nl-llm-agent-recur-supervised--loss-one
                      model geometry trajectory start options)))
                (unless (nl-llm-agent-recur-supervised--finite-p
                         (aref (photon-tensor-data (pav-value loss)) 0))
                  (error "recurrent supervised loss is non-finite"))
                (photon-autograd-backward loss)
                (unless (nl-llm-agent-recur-supervised--all-finite-p parameters)
                  (error "recurrent supervised gradients are non-finite"))
                (photon-autograd-sgd parameters (plist-get options :lr))
                (unless (nl-llm-agent-recur-supervised--all-finite-p parameters)
                  (error "recurrent supervised update is non-finite")))
              (setq index (1+ index)))))
        (setq epoch (1+ epoch))))
    (list :backend 'cpu :model-family 'recurrent-depth
          :r (plist-get options :r) :k (plist-get options :k)
          :s0-seed (plist-get options :s0-seed)
          :steps (* (plist-get options :epochs)
                    (plist-get plan :examples))
          :examples (plist-get plan :examples)
          :completion-tokens (plist-get plan :completion-tokens)
          :dataset-sha256
          (copy-sequence (plist-get plan :dataset-sha256))
          :loss-before loss-before
          :loss-after
          (nl-llm-agent-recur-supervised--loss-plan
           model geometry plan options))))

(provide 'nl-llm-agent-recur-supervised)
;;; nl-llm-agent-recur-supervised.el ends here
