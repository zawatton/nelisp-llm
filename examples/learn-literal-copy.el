;;; learn-literal-copy.el --- learn exact short literal copying on the GPU -*- lexical-binding: t; -*-

;; This is deliberately a small, closed supervised probe.  It does not use the
;; agent grammar or the model-policy renderer: the native decoder below sees
;; only logits and is allowed to choose any of the 256 byte ids.
;;
;; Run from this directory with:
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp \
;;     -l examples/learn-literal-copy.el

(defvar nl-llm-learn-literal-copy-no-run nil
  "When non-nil, load this file without starting the GPU experiment.")

(let ((here (file-name-directory (or load-file-name buffer-file-name))))
  (add-to-list 'load-path (expand-file-name "../lisp" here))
  (add-to-list 'load-path (expand-file-name "../../nelisp-photon/lisp" here)))

(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-agent-artifact)
(require 'nl-llm-agent-improve)
(require 'nl-llm-agent-ondevice)
(require 'nl-llm-agent-supervised)
(require 'nl-llm-agent-tokenizer)
(require 'nl-llm-decode)
(require 'nl-llm-ckpt)
(require 'nl-llm-gpu)

(defconst nl-llm-learn-literal-copy-alphabet "abc")
(defconst nl-llm-learn-literal-copy-tokenizer "utf8-byte-v1")
(defconst nl-llm-learn-literal-copy-dim 32)
(defconst nl-llm-learn-literal-copy-ff 64)
(defconst nl-llm-learn-literal-copy-vocab 256)
(defconst nl-llm-learn-literal-copy-blocks 1)
(defconst nl-llm-learn-literal-copy-heads 1)
(defconst nl-llm-learn-literal-copy-sequence 32)
(defconst nl-llm-learn-literal-copy-max-decode 8)
(defconst nl-llm-learn-literal-copy-learning-rate 0.003)
(defconst nl-llm-learn-literal-copy-epochs 32)

(defun nl-llm-learn-literal-copy--strings-of-length (length)
  "Return alphabet strings of LENGTH in lexical order."
  (if (= length 0)
      '("")
    (apply #'append
           (mapcar
            (lambda (char)
              (mapcar (lambda (tail) (concat (string char) tail))
                      (nl-llm-learn-literal-copy--strings-of-length
                       (1- length))))
            (string-to-list nl-llm-learn-literal-copy-alphabet)))))

;;;###autoload
(defun nl-llm-learn-literal-copy-literals ()
  "Return the 39 literals, length first and lexical within each length."
  (apply #'append
         (mapcar #'nl-llm-learn-literal-copy--strings-of-length '(1 2 3))))

(defun nl-llm-learn-literal-copy--example (index literal)
  (list :index index
        :literal literal
        :prompt (format "COPY: %s\nOUTPUT:\n" literal)
        :completion (concat literal "\n")))

;;;###autoload
(defun nl-llm-learn-literal-copy-dataset ()
  "Return the deterministic all/train/dev split as detached vectors.

Global indices divisible by five are the eight development examples.  Every
other index is a training example; no example is duplicated between the two
sets."
  (let ((all nil) (train nil) (dev nil) (index 0))
    (dolist (literal (nl-llm-learn-literal-copy-literals))
      (let ((example (nl-llm-learn-literal-copy--example index literal)))
        (push example all)
        (if (= (% index 5) 0)
            (push example dev)
          (push example train)))
      (setq index (1+ index)))
    (list :all (vconcat (nreverse all))
          :train (vconcat (nreverse train))
          :dev (vconcat (nreverse dev)))))

(defun nl-llm-learn-literal-copy--finite-number-p (value)
  (and (numberp value) (= value value)
       (< (abs value) 1.7976931348623157e+308)))

(defun nl-llm-learn-literal-copy--finite-vector-p (values)
  (and (vectorp values)
       (cl-loop for value across values
                always (nl-llm-learn-literal-copy--finite-number-p value))))

(defun nl-llm-learn-literal-copy--argmax (logits)
  "Return the unrestricted argmax id in LOGITS, including ids 0 and 255."
  (unless (and (= (length logits) nl-llm-learn-literal-copy-vocab)
               (nl-llm-learn-literal-copy--finite-vector-p logits))
    (error "literal-copy decoder received invalid logits"))
  (let ((best 0) (value (aref logits 0)))
    (dotimes (id (1- (length logits)))
      (let ((candidate (1+ id)))
        (when (> (aref logits candidate) value)
          (setq best candidate value (aref logits candidate)))))
    best))

;;;###autoload
(defun nl-llm-learn-literal-copy-greedy-decode
    (prompt-ids step-function &optional max-generated)
  "Decode PROMPT-IDS with unrestricted byte argmax STEP-FUNCTION.

STEP-FUNCTION receives each consumed token id and returns a fresh 256-logit
vector for the next token.  Generation stops only on byte id 10 or after eight
generated ids (or MAX-GENERATED when supplied).  No expected answer or
candidate alphabet is passed to, or consulted by, this decoder.  Return a
plist containing numeric `:ids' and boolean `:terminated'."
  (unless (and (listp prompt-ids) (functionp step-function))
    (error "literal-copy decoder requires prompt ids and a step function"))
  (let ((logits nil)
        (generated nil)
        (limit (or max-generated nl-llm-learn-literal-copy-max-decode))
        (terminated nil))
    (unless (and (integerp limit) (> limit 0))
      (error "literal-copy decoder max-generated must be positive"))
    (dolist (token prompt-ids)
      (unless (and (integerp token) (<= 0 token) (< token 256))
        (error "literal-copy prompt has invalid byte id %S" token))
      (setq logits (funcall step-function token)))
    (unless logits
      (error "literal-copy decoder requires a non-empty prompt"))
    (catch 'done
      (dotimes (_ limit)
        (let ((token (nl-llm-learn-literal-copy--argmax logits)))
          (push token generated)
          (if (= token 10)
              (progn (setq terminated t) (throw 'done nil))
            (setq logits (funcall step-function token))))))
    (list :ids (vconcat (nreverse generated)) :terminated terminated)))

(defun nl-llm-learn-literal-copy--native-model (model)
  "Export trainable MODEL and validate it as a raw decoder checkpoint."
  (let ((exported (nl-llm-agent-artifact-export-pav model)))
    (nl-llm-agent-artifact--checkpoint-model
     (append (list :format nl-llm-ckpt-format) exported)
     "literal-copy native decoder")))

;;;###autoload
(defun nl-llm-learn-literal-copy-native-decode (model prompt-ids)
  "Run raw native greedy decode for MODEL and numeric PROMPT-IDS.

The returned plist has `:ids' and `:terminated'; it intentionally has no
expected-answer or grammar field.  Model output is read through 64-position
KV caches and all 256 output ids are eligible at every step."
  (let* ((native (nl-llm-learn-literal-copy--native-model model))
         (blocks (plist-get native :blocks))
         (dim (plist-get native :dim))
         (heads (plist-get native :heads))
         (kvh (plist-get native :kvh))
         (caches (mapcar (lambda (_block)
                           (nl-llm-dcache-new 64 dim heads kvh))
                         blocks))
         (step (lambda (token)
                 (nl-llm-decode-step
                  token blocks caches
                  (plist-get native :wte) (plist-get native :lnfg)
                  (plist-get native :bh) dim nil (plist-get native :wh)))))
    (nl-llm-learn-literal-copy-greedy-decode prompt-ids step)))

(defun nl-llm-learn-literal-copy--examples-vector (dataset key)
  (apply #'vector
         (mapcar (lambda (example)
                   (list :prompt (plist-get example :prompt)
                         :completion (plist-get example :completion)))
                 (append (plist-get dataset key) nil))))

(defun nl-llm-learn-literal-copy--flat-weights (model)
  (apply #'vconcat
         (mapcar (lambda (parameter)
                   (copy-sequence
                    (photon-tensor-data (pav-value parameter))))
                 (nl-llm-agent--p5-params model))))

(defun nl-llm-learn-literal-copy--sha256 (value)
  (let ((print-length nil) (print-level nil) (print-circle nil)
        (print-escape-nonascii t) (float-output-format nil))
    (secure-hash 'sha256 (prin1-to-string value))))

(defun nl-llm-learn-literal-copy--model-hash (model)
  "Hash MODEL geometry and finite trainable weights in canonical form."
  (let ((weights (nl-llm-learn-literal-copy--flat-weights model)))
    (unless (cl-every #'nl-llm-learn-literal-copy--finite-number-p weights)
      (error "model weights are non-finite"))
    (nl-llm-learn-literal-copy--sha256
     (list :geometry
           (list :dim (plist-get model :dim)
                 :ff (plist-get model :ff)
                 :vocab (plist-get model :vocab)
                 :heads (plist-get model :heads)
                 :nblocks (plist-get model :nblocks)
                 :tokenizer (plist-get model :tokenizer))
           :weights (append weights nil)))))

(defun nl-llm-learn-literal-copy--loss-finite (loss where)
  (unless (nl-llm-learn-literal-copy--finite-number-p loss)
    (error "%s loss is non-finite: %S" where loss))
  loss)

(defun nl-llm-learn-literal-copy--score (model examples label)
  "Return a structured exact-copy report for EXAMPLES using MODEL."
  (let ((ok 0) (count (length examples)))
    (let ((cases nil))
      (dolist (example (append examples nil))
        (let* ((index (plist-get example :index))
               (literal (plist-get example :literal))
             (prompt-ids
              (nl-llm-agent-tokenizer-encode
               (plist-get example :prompt) nl-llm-learn-literal-copy-tokenizer))
             (expected
              (append
               (nl-llm-agent-tokenizer-encode
                (plist-get example :completion)
                nl-llm-learn-literal-copy-tokenizer)
               nil))
             (decoded
              (nl-llm-learn-literal-copy-native-decode model prompt-ids))
             (actual (vconcat (plist-get decoded :ids)))
             (expected-vector (vconcat expected))
             (terminated (plist-get decoded :terminated))
             (exact (and terminated (equal actual expected-vector))))
          (when exact (setq ok (1+ ok)))
          (push (list :index index :literal literal
                      :expected expected-vector :ids actual
                      :terminated terminated :exact exact)
                cases)))
      (list :label label :cases (vconcat (nreverse cases))
            :passed ok :total count :score (/ (float ok) count)))))

;;;###autoload
(defun nl-llm-learn-literal-copy-run ()
  "Run the bounded literal-copy GPU training experiment and return its report."
  (let* ((dataset (nl-llm-learn-literal-copy-dataset))
         (train (plist-get dataset :train))
         (dev (plist-get dataset :dev))
         (train-examples
          (nl-llm-learn-literal-copy--examples-vector dataset :train))
         (encoded
          (nl-llm-agent-supervised-encode
           train-examples nl-llm-learn-literal-copy-tokenizer))
         (max-tokens
          (apply #'max
                 (mapcar #'length (plist-get encoded :trajectories))))
         (model
          (nl-llm-agent-improve-model
           nl-llm-learn-literal-copy-dim nl-llm-learn-literal-copy-ff
           nl-llm-learn-literal-copy-vocab nl-llm-learn-literal-copy-blocks
           nl-llm-learn-literal-copy-heads nl-llm-learn-literal-copy-tokenizer))
         (identity model)
         (settings
          (list :alphabet nl-llm-learn-literal-copy-alphabet
                :literals (length (plist-get dataset :all))
                :train (length train) :dev (length dev)
                :vocab nl-llm-learn-literal-copy-vocab
                :dim nl-llm-learn-literal-copy-dim
                :ff nl-llm-learn-literal-copy-ff
                :blocks nl-llm-learn-literal-copy-blocks
                :heads nl-llm-learn-literal-copy-heads
                :sequence nl-llm-learn-literal-copy-sequence
                :learning-rate nl-llm-learn-literal-copy-learning-rate
                :epochs nl-llm-learn-literal-copy-epochs
                :optimizer 'adam :transfer-mode 'compact
                :max-tokens max-tokens)))
    (when (> max-tokens
             nl-llm-learn-literal-copy-sequence)
      (error "literal-copy encoded sequence exceeds GPU sequence"))
    ;; The GPU backend changes tensor operations globally.  Keep all CPU loss
    ;; and native decode measurements outside the enabled-GPU dynamic extent.
    (message "literal-copy beforeeval")
    (let* ((before-hash (nl-llm-learn-literal-copy--model-hash model))
           (before-loss
            (nl-llm-learn-literal-copy--loss-finite
             (nl-llm-agent-supervised-loss model train-examples) "before"))
           (before-train
            (nl-llm-learn-literal-copy--score model train "before-train"))
           (before-dev
            (nl-llm-learn-literal-copy--score model dev "before-dev"))
           (before-eval-hash (nl-llm-learn-literal-copy--model-hash model))
           (gpu-enabled nil) (context nil) (training-result nil))
      (unless (equal before-hash before-eval-hash)
        (error "before evaluation mutated trainable weights"))
      (message "literal-copy training start: %d steps"
               (* nl-llm-learn-literal-copy-epochs (length train)))
      (unwind-protect
          (progn
            (setq gpu-enabled (nl-llm-gpu-enable))
            (unless gpu-enabled
              (error "literal-copy training requires a Vulkan GPU"))
            (setq context
                  (nl-llm-agent-ondevice-from-model
                   model nl-llm-learn-literal-copy-sequence
                   nl-llm-learn-literal-copy-learning-rate
                   :optimizer 'adam :loss-mode 'completion :transfer-mode 'compact))
            (setq training-result
                  (nl-llm-agent-ondevice-train
                   context (plist-get encoded :trajectories)
                   nl-llm-learn-literal-copy-epochs
                   :loss-starts (plist-get encoded :loss-starts)
                   :after-step
                   (lambda (_ctx completed total)
                     (when (and (> completed 0)
                                (= (% completed (length train)) 0)
                                (= (% (/ completed (length train)) 4) 0))
                       (message "literal-copy train epoch=%d/%d steps=%d/%d"
                                (/ completed (length train))
                                nl-llm-learn-literal-copy-epochs
                                completed total)))))
            (nl-llm-agent-ondevice-sync context)
            (nl-llm-agent-ondevice-free context)
            (setq context nil)
            ;; Disable exactly once before any CPU loss or native inference.
            (nl-llm-gpu-disable)
            (setq gpu-enabled nil))
        (when context
          (nl-llm-agent-ondevice-free context))
        (when gpu-enabled
          (nl-llm-gpu-disable)))
      (message "literal-copy aftereval")
      (let* ((after-hash (nl-llm-learn-literal-copy--model-hash model))
             (after-loss
              (nl-llm-learn-literal-copy--loss-finite
               (nl-llm-agent-supervised-loss model train-examples) "after"))
             (after-train
              (nl-llm-learn-literal-copy--score model train "after-train"))
             (after-dev
              (nl-llm-learn-literal-copy--score model dev "after-dev"))
             (after-eval-hash (nl-llm-learn-literal-copy--model-hash model))
             (identity-preserved
              (and (eq model identity)
                   (equal before-hash before-eval-hash)
                   (equal after-hash after-eval-hash))))
        (unless identity-preserved
          (error "literal-copy evaluation did not preserve model identity"))
        (list :format "nl-llm-literal-copy-report-v1"
              :settings settings
              :dataset-sha256 (plist-get encoded :dataset-sha256)
              :model-before-sha256 before-hash
              :model-after-sha256 after-hash
              :before (list :train before-train :dev before-dev)
              :after (list :train after-train :dev after-dev)
              :loss-before before-loss :loss-after after-loss
              :weightchanged (not (equal before-hash after-hash))
              :modelidentitypreserved identity-preserved
              :steps training-result)))))

(provide 'learn-literal-copy)

(unless nl-llm-learn-literal-copy-no-run
  (princ (format "%S\n" (nl-llm-learn-literal-copy-run))))

;;; learn-literal-copy.el ends here
