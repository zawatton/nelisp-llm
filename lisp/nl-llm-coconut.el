;;; nl-llm-coconut.el --- continuous latent reasoning on CPU autograd  -*- lexical-binding: t; -*-

;; Coconut replaces leading language reasoning steps with continuous thoughts.
;; Each thought is the final-RMSNorm hidden row from a full prefix pass, fed
;; directly back as the next input embedding.  This module supplies the row
;; autograd operations, a small deterministic transformer, the staged forward
;; and loss paths, greedy generation, and a synthetic curriculum task.

;;; Code:

(require 'cl-lib)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-autograd)

;;;###autoload
(defun nl-llm-ag-slice-rows (x r0 nrows)
  "Extract NROWS rows at R0 from autograd tensor X.
X must have shape (M x N).  The result has shape (NROWS x N), and its
backward pass scatters the upstream gradient into the selected rows of X."
  (let* ((xv (pav-value x))
         (shape (photon-tensor-shape xv))
         (rows (car shape))
         (cols (nth 1 shape)))
    (when (or (< r0 0) (< nrows 0) (> (+ r0 nrows) rows))
      (error "nl-llm-ag-slice-rows: range %d+%d outside %d rows"
             r0 nrows rows))
    (let* ((xd (photon-tensor-data xv))
           (out (make-vector (* nrows cols) 0.0))
           (count (* nrows cols))
           (start (* r0 cols))
           (i 0))
      (while (< i count)
        (aset out i (aref xd (+ start i)))
        (setq i (1+ i)))
      (photon-autograd--record
       (photon-tensor (list nrows cols) out)
       (lambda (g)
         (let* ((gd (photon-tensor-data g))
                (dx (make-vector (* rows cols) 0.0))
                (j 0))
           (while (< j count)
             (aset dx (+ start j) (aref gd j))
             (setq j (1+ j)))
           (photon-autograd--addgrad
            x (photon-tensor (list rows cols) dx))))))))

;;;###autoload
(defun nl-llm-ag-concat-rows (tensors)
  "Stack a non-empty list of autograd TENSORS by rows.
Every input must be rank two with the same column count.  The backward pass
sends each contiguous row slice of the upstream gradient to its input."
  (unless tensors
    (error "nl-llm-ag-concat-rows: empty tensor list"))
  (let* ((first-shape (photon-tensor-shape (pav-value (car tensors))))
         (cols (nth 1 first-shape))
         (rows 0)
         (rest tensors))
    (while rest
      (let ((shape (photon-tensor-shape (pav-value (car rest)))))
        (unless (and (= (length shape) 2) (= (nth 1 shape) cols))
          (error "nl-llm-ag-concat-rows: incompatible shape %S" shape))
        (setq rows (+ rows (car shape))))
      (setq rest (cdr rest)))
    (let ((out (make-vector (* rows cols) 0.0))
          (row-offset 0))
      (dolist (tensor tensors)
        (let* ((value (pav-value tensor))
               (shape (photon-tensor-shape value))
               (input-rows (car shape))
               (data (photon-tensor-data value))
               (count (* input-rows cols))
               (dst (* row-offset cols))
               (i 0))
          (while (< i count)
            (aset out (+ dst i) (aref data i))
            (setq i (1+ i)))
          (setq row-offset (+ row-offset input-rows))))
      (photon-autograd--record
       (photon-tensor (list rows cols) out)
       (lambda (g)
         (let ((gd (photon-tensor-data g))
               (back-row-offset 0))
           (dolist (tensor tensors)
             (let* ((shape (photon-tensor-shape (pav-value tensor)))
                    (input-rows (car shape))
                    (count (* input-rows cols))
                    (src (* back-row-offset cols))
                    (dx (make-vector count 0.0))
                    (i 0))
               (while (< i count)
                 (aset dx i (aref gd (+ src i)))
                 (setq i (1+ i)))
               (photon-autograd--addgrad
                tensor (photon-tensor (list input-rows cols) dx))
               (setq back-row-offset (+ back-row-offset input-rows))))))))))

(defun nl-llm-coconut--parameter (shape seed scale)
  "Return a hash-initialized autograd parameter of SHAPE.
SEED selects a deterministic stream and SCALE bounds the values."
  (let ((size 1))
    (dolist (dim shape)
      (setq size (* size dim)))
    (let ((data (make-vector size 0.0))
          (i 0))
      (while (< i size)
        (aset data i
              (* scale 2.0
                 (- (/ (float
                        (mod (+ (* (1+ i) 2654435761)
                                (* (1+ seed) 40503))
                              65536))
                       65536.0)
                    0.5)))
        (setq i (1+ i)))
      (photon-autograd-const (photon-tensor shape data)))))

(defun nl-llm-coconut--constant (n value)
  "Return a length-N autograd parameter filled with VALUE."
  (photon-autograd-const
   (photon-tensor (list n) (make-vector n (float value)))))

(defun nl-llm-coconut--block-new (dim ff heads kv-heads seed)
  "Build one deterministic SwiGLU transformer block.
DIM, FF, HEADS, and KV-HEADS define its shape; SEED selects its weights."
  (let* ((head-dim (/ dim heads))
         (kv-dim (* kv-heads head-dim))
         (scale (/ 1.0 (sqrt (float dim)))))
    (list :ln1g (nl-llm-coconut--constant dim 1.0)
          :wq (nl-llm-coconut--parameter (list dim dim) (+ seed 1) scale)
          :bq (nl-llm-coconut--constant dim 0.0)
          :wk (nl-llm-coconut--parameter (list kv-dim dim) (+ seed 2) scale)
          :bk (nl-llm-coconut--constant kv-dim 0.0)
          :wv (nl-llm-coconut--parameter (list kv-dim dim) (+ seed 3) scale)
          :bv (nl-llm-coconut--constant kv-dim 0.0)
          :wo (nl-llm-coconut--parameter (list dim dim) (+ seed 4) scale)
          :bo (nl-llm-coconut--constant dim 0.0)
          :ln2g (nl-llm-coconut--constant dim 1.0)
          :wg (nl-llm-coconut--parameter (list ff dim) (+ seed 5) scale)
          :bg (nl-llm-coconut--constant ff 0.0)
          :wu (nl-llm-coconut--parameter (list ff dim) (+ seed 6) scale)
          :bu (nl-llm-coconut--constant ff 0.0)
          :wd (nl-llm-coconut--parameter (list dim ff) (+ seed 7) scale)
          :bd (nl-llm-coconut--constant dim 0.0))))

;;;###autoload
(cl-defun nl-llm-coconut-model-new
    (&key (vocab 12) (dim 16) (heads 2) (kv-heads 1) (ff 16)
          (nblocks 2) n-blocks (seed 1) bot eot)
  "Build a small hash-seeded Coconut model of autograd parameters.
VOCAB, DIM, HEADS, KV-HEADS, FF, and NBLOCKS (or N-BLOCKS) set the
transformer shape.  SEED deterministically initializes every weight.  BOT and
EOT default to the final two vocabulary ids.  The returned plist contains
:wte, :blocks, :lnfg, :wh, :bh, :dim, :heads, :kv-heads, :bot, and :eot."
  (let ((block-count (or n-blocks nblocks))
        (bot-id (or bot (- vocab 2)))
        (eot-id (or eot (1- vocab))))
    (unless (and (> vocab 1) (> dim 0) (> heads 0) (> kv-heads 0)
                 (= (% dim heads) 0) (= (% heads kv-heads) 0)
                 (> ff 0) (> block-count 0)
                 (>= bot-id 0) (< bot-id vocab)
                 (>= eot-id 0) (< eot-id vocab))
      (error "nl-llm-coconut-model-new: invalid model shape or marker ids"))
    (let ((blocks nil)
          (i 0)
          (scale (/ 1.0 (sqrt (float dim)))))
      (while (< i block-count)
        (push (nl-llm-coconut--block-new
               dim ff heads kv-heads (+ seed (* 20 (1+ i))))
              blocks)
        (setq i (1+ i)))
      (list :wte (nl-llm-coconut--parameter (list vocab dim) seed scale)
            :blocks (nreverse blocks)
            :lnfg (nl-llm-coconut--constant dim 1.0)
            :wh (nl-llm-coconut--parameter (list vocab dim) (+ seed 9001) scale)
            :bh (nl-llm-coconut--constant vocab 0.0)
            :dim dim :heads heads :kv-heads kv-heads
            :vocab vocab :ff ff :nblocks block-count
            :bot bot-id :eot eot-id))))

;;;###autoload
(defun nl-llm-coconut-params (model)
  "Return every trainable autograd parameter in MODEL in stable order."
  (let ((params (list (plist-get model :wte))))
    (dolist (block (plist-get model :blocks))
      (dolist (key '(:ln1g :wq :bq :wk :bk :wv :bv :wo :bo
                     :ln2g :wg :bg :wu :bu :wd :bd))
        (setq params (append params (list (plist-get block key))))))
    (append params
            (list (plist-get model :lnfg)
                  (plist-get model :wh)
                  (plist-get model :bh)))))

(defun nl-llm-coconut--hidden (model embeddings)
  "Run MODEL blocks and final RMSNorm over EMBEDDINGS; return hidden rows."
  (let ((x embeddings)
        (heads (plist-get model :heads))
        (kv-heads (plist-get model :kv-heads))
        (rope-base (plist-get model :rope-base)))
    (dolist (block (plist-get model :blocks))
      (setq x (nl-llm-ag-block x block heads kv-heads rope-base)))
    (nl-llm-ag-rmsnorm x (plist-get model :lnfg))))

(defun nl-llm-coconut--forward (model question c suffix zero-thoughts)
  "Internal Coconut forward, optionally replacing latent rows with zeros.
MODEL, QUESTION, C, and SUFFIX have the public forward semantics.
ZERO-THOUGHTS is used only by verification to ablate the feedback vectors."
  (when (< c 0)
    (error "nl-llm-coconut-forward: C must be non-negative"))
  (photon-autograd-reset-tape)
  (let* ((dim (plist-get model :dim))
         (wte (plist-get model :wte))
         (bot (plist-get model :bot))
         (eot (plist-get model :eot)))
    (if (= c 0)
        (let* ((tokens (append question (list bot eot) suffix))
               (embeddings (photon-autograd-embedding wte tokens dim))
               (hidden (nl-llm-coconut--hidden model embeddings)))
          (photon-autograd-linear
           hidden (plist-get model :wh) (plist-get model :bh)))
      (let ((prefix (photon-autograd-embedding
                     wte (append question (list bot)) dim))
            (i 0))
        (while (< i c)
          (let* ((hidden (nl-llm-coconut--hidden model prefix))
                 (rows (car (photon-tensor-shape (pav-value hidden))))
                 (thought (nl-llm-ag-slice-rows hidden (1- rows) 1)))
            (when zero-thoughts
              (setq thought
                    (photon-autograd-const
                     (photon-tensor (list 1 dim) (make-vector dim 0.0)))))
            (setq prefix (nl-llm-ag-concat-rows (list prefix thought))))
          (setq i (1+ i)))
        (let* ((tail (photon-autograd-embedding
                      wte (cons eot (append suffix nil)) dim))
               (embeddings (nl-llm-ag-concat-rows (list prefix tail)))
               (hidden (nl-llm-coconut--hidden model embeddings)))
          (photon-autograd-linear
           hidden (plist-get model :wh) (plist-get model :bh)))))))

;;;###autoload
(defun nl-llm-coconut-forward (model question c suffix)
  "Run Coconut MODEL on QUESTION, C continuous thoughts, and token SUFFIX.
QUESTION and SUFFIX are token-id lists.  The result is a logits autograd value
with rows for QUESTION, BOT, C thoughts, EOT, and SUFFIX.  Each thought is the
post-final-RMSNorm hidden row from the preceding full-prefix pass."
  (nl-llm-coconut--forward model question c suffix nil))

;;;###autoload
(defun nl-llm-coconut-loss (model question c suffix)
  "Return Coconut cross-entropy for EOT followed by SUFFIX.
Loss-bearing logit rows begin at the final thought (or BOT when C is zero), so
question and continuous-thought positions themselves carry no targets."
  (let* ((logits (nl-llm-coconut-forward model question c suffix))
         (first-row (+ (length question) c))
         (targets-list (cons (plist-get model :eot) (append suffix nil)))
         (loss-logits (nl-llm-ag-slice-rows
                       logits first-row (length targets-list))))
    (photon-autograd-softmax-ce
     loss-logits (apply #'vector targets-list))))

(defun nl-llm-coconut--argmax-row (logits row vocab)
  "Return the index of the largest value in ROW of LOGITS over VOCAB columns."
  (let* ((data (photon-tensor-data (pav-value logits)))
         (base (* row vocab))
         (best 0)
         (best-value (aref data base))
         (i 1))
    (while (< i vocab)
      (let ((value (aref data (+ base i))))
        (when (> value best-value)
          (setq best i best-value value)))
      (setq i (1+ i)))
    best))

;;;###autoload
(defun nl-llm-coconut-generate (model question c max-new)
  "Greedily generate at most MAX-NEW language tokens from Coconut MODEL.
The model first performs C continuous thoughts, then consumes an explicit EOT
marker.  Returned tokens exclude that marker and stop before a generated EOT."
  (let ((tokens nil)
        (vocab (or (plist-get model :vocab)
                   (car (photon-tensor-shape
                         (pav-value (plist-get model :wh))))))
        (eot (plist-get model :eot))
        (done nil)
        (n 0))
    (while (and (< n max-new) (not done))
      (let* ((logits (nl-llm-coconut-forward model question c tokens))
             (rows (car (photon-tensor-shape (pav-value logits))))
             (token (nl-llm-coconut--argmax-row logits (1- rows) vocab)))
        (if (= token eot)
            (setq done t)
          (setq tokens (append tokens (list token)))
          (setq n (1+ n)))))
    tokens))

;;;###autoload
(defun nl-llm-coconut-stage-example (item k c)
  "Map curriculum ITEM to a (QUESTION . SUFFIX) pair for stage K.
ITEM is a plist with :q, :steps, and :a.  Stage K removes its first K language
reasoning steps; C is validated because it is the thoughts-per-step curriculum
setting, although it does not alter the token pair itself."
  (when (or (< k 0) (< c 0))
    (error "nl-llm-coconut-stage-example: K and C must be non-negative"))
  (let* ((question (append (plist-get item :q) nil))
         (steps (plist-get item :steps))
         (skip (min k (length steps)))
         (suffix (append (nthcdr skip steps)
                         (list (plist-get item :a)))))
    (cons question suffix)))

(defun nl-llm-coconut--loss-value (loss)
  "Return the scalar float stored in autograd LOSS."
  (aref (photon-tensor-data (pav-value loss)) 0))

;;;###autoload
(cl-defun nl-llm-coconut-train
    (model data &key (stages 2) (epochs 3) (lr 0.4) (c 1) trace)
  "Train MODEL on curriculum DATA with SGD and return epoch mean losses.
Stages zero through STAGES are run in order.  At stage K, the first K language
steps are replaced by K*C continuous thoughts.  EPOCHS and LR control each
stage.  The return value is one loss-trajectory list per stage.  If TRACE is
non-nil, call it with (STAGE EPOCH MEAN-LOSS) after every epoch."
  (let ((params (nl-llm-coconut-params model))
        (stage-results nil)
        (stage 0))
    (while (<= stage stages)
      (let ((epoch-results nil)
            (epoch 0))
        (while (< epoch epochs)
          (let ((total 0.0)
                (count 0))
            (dolist (item data)
              (let* ((example (nl-llm-coconut-stage-example item stage c))
                     (loss (nl-llm-coconut-loss
                            model (car example) (* stage c) (cdr example))))
                (setq total (+ total (nl-llm-coconut--loss-value loss)))
                (setq count (1+ count))
                (photon-autograd-zero-grad params)
                (photon-autograd-backward loss)
                (photon-autograd-sgd params lr)))
            (let ((mean (/ total (float count))))
              (push mean epoch-results)
              (when trace
                (funcall trace stage epoch mean))))
          (setq epoch (1+ epoch)))
        (push (nreverse epoch-results) stage-results))
      (setq stage (1+ stage)))
    (nreverse stage-results)))

;;;###autoload
(defun nl-llm-coconut-task-chain-add (k n seed)
  "Return N seeded K-digit chain-add curriculum items.
Each item is (:q DIGITS :steps RUNNING-SUMS :a ANSWER).  RUNNING-SUMS contains
K-1 sums modulo ten, and ANSWER is the final sum.  Digit ids are 0 through 9;
model marker ids are therefore conventionally BOT=10 and EOT=11."
  (when (< k 1)
    (error "nl-llm-coconut-task-chain-add: K must be positive"))
  (random (format "nl-llm-coconut-chain-add-%S" seed))
  (let ((items nil)
        (item-index 0))
    (while (< item-index n)
      (let ((digits nil)
            (digit-index 0))
        (while (< digit-index k)
          (push (random 10) digits)
          (setq digit-index (1+ digit-index)))
        (setq digits (nreverse digits))
        (let ((sum (car digits))
              (remaining (cdr digits))
              (steps nil))
          (while remaining
            (setq sum (% (+ sum (car remaining)) 10))
            (push sum steps)
            (setq remaining (cdr remaining)))
          (push (list :q digits :steps (nreverse steps) :a sum) items)))
      (setq item-index (1+ item-index)))
    (nreverse items)))

(provide 'nl-llm-coconut)
;;; nl-llm-coconut.el ends here
