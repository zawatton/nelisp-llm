;;; learn-copy-curriculum.el --- deterministic COPY curriculum pretraining -*- lexical-binding: t; -*-

;; A bounded pretraining probe over synthetic ASCII COPY examples.  The
;; frozen literal-copy decoder and model export are reused; this file changes
;; only the dataset, initializer, and training/evaluation orchestration.

;;; Code:

(require 'cl-lib)

(defconst nl-llm-learn-copy-curriculum-source-sha256
  "ac0742137067ad93eccdafd9e4182ffa56744480f4e2fe9e75edc4770994d459")
(defconst nl-llm-learn-copy-curriculum-format
  "nl-llm-copy-curriculum-report-v1")
(defconst nl-llm-learn-copy-curriculum-data-format
  "nl-copy-curriculum-data-v1")
(defconst nl-llm-learn-copy-curriculum-seed 608135816)
(defconst nl-llm-learn-copy-curriculum-initializer-seed 439041101)
(defconst nl-llm-learn-copy-curriculum-alphabet
  "abcdefghijklmnopqrstuvwxyz0123456789/._-: ")
(defconst nl-llm-learn-copy-curriculum-lengths '(1 2 3 4 8 12 16 24))
(defconst nl-llm-learn-copy-curriculum-per-length 20)
(defconst nl-llm-learn-copy-curriculum-tokenizer "utf8-byte-v1")
(defconst nl-llm-learn-copy-curriculum-dim 32)
(defconst nl-llm-learn-copy-curriculum-ff 64)
(defconst nl-llm-learn-copy-curriculum-vocab 256)
(defconst nl-llm-learn-copy-curriculum-blocks 1)
(defconst nl-llm-learn-copy-curriculum-heads 1)
(defconst nl-llm-learn-copy-curriculum-sequence 64)
(defconst nl-llm-learn-copy-curriculum-learning-rate 0.003)
(defconst nl-llm-learn-copy-curriculum-epochs 32)
(defconst nl-llm-learn-copy-curriculum-max-generated 32)
(defconst nl-llm-learn-copy-curriculum-uint32-mask #xffffffff)
(defvar nl-llm-learn-copy-curriculum-auto-run nil
  "When non-nil, loading this example runs the GPU probe.")

;; The frozen file has a top-level runner.  Declare its guard before loading
;; so the source can be reused without training or changing global state.
(defvar nl-llm-learn-literal-copy-no-run nil)

(defun nl-llm-learn-copy-curriculum--file-sha256 (path)
  "Return byte SHA-256 digest of PATH without text decoding."
  (with-temp-buffer
    (set-buffer-multibyte nil)
    (insert-file-contents-literally path)
    (secure-hash 'sha256 (current-buffer))))

(let* ((here (file-name-directory (or load-file-name buffer-file-name)))
       (source (expand-file-name "learn-literal-copy.el" here)))
  (unless (equal (nl-llm-learn-copy-curriculum--file-sha256 source)
                 nl-llm-learn-copy-curriculum-source-sha256)
    (error "frozen literal-copy source SHA mismatch: %s"
           (nl-llm-learn-copy-curriculum--file-sha256 source)))
  (let ((nl-llm-learn-literal-copy-no-run t))
    (load source nil nil t)))

(require 'cl-lib)
(require 'nl-llm-agent-initialization)
(require 'nl-llm-agent-supervised)
(require 'nl-llm-agent-tokenizer)
(require 'nl-llm-agent-ondevice)
(require 'nl-llm-inference-runtime)
(require 'nl-llm-gpu)

(declare-function nl-llm-agent-initialization-create
                  "nl-llm-agent-initialization" (&rest keys))
(declare-function nl-llm-agent-supervised-encode
                  "nl-llm-agent-supervised" (examples &optional tokenizer))
(declare-function nl-llm-agent-tokenizer-encode
                  "nl-llm-agent-tokenizer" (text &optional identifier))
(declare-function nl-llm-agent-ondevice-from-model
                  "nl-llm-agent-ondevice" (model seq lr &rest keys))
(declare-function nl-llm-agent-ondevice-train
                  "nl-llm-agent-ondevice" (ctx trajs epochs &rest keys))
(declare-function nl-llm-agent-ondevice-sync
                  "nl-llm-agent-ondevice" (ctx))
(declare-function nl-llm-agent-ondevice-free
                  "nl-llm-agent-ondevice" (ctx))
(declare-function nl-llm-gpu-available-p "nl-llm-gpu" ())
(declare-function nl-llm-gpu-enable "nl-llm-gpu" ())
(declare-function nl-llm-gpu-disable "nl-llm-gpu" ())
(declare-function nl-llm-inference-runtime-prepare
                  "nl-llm-inference-runtime" (&optional mode))
(declare-function nl-llm-dcache-new "nl-llm-decode"
                  (max-seq dim heads kvh))
(declare-function nl-llm-decode-step "nl-llm-decode"
                  (token blocks caches wte lnfg bh dim rope wh))
(declare-function nl-llm-learn-literal-copy--model-hash
                  "learn-literal-copy" (model))
(declare-function nl-llm-learn-literal-copy--native-model
                  "learn-literal-copy" (model))
(declare-function nl-llm-learn-literal-copy-greedy-decode
                  "learn-literal-copy"
                  (prompt-ids step-function &optional max-generated))

(defun nl-llm-learn-copy-curriculum--xorshift32 (state)
  "Return next local uint32 xorshift value from STATE."
  (let ((mask nl-llm-learn-copy-curriculum-uint32-mask))
    (setq state (logand mask (logxor state (ash state 13))))
    (setq state (logand mask (logxor state (ash state -17))))
    (logand mask (logxor state (ash state 5)))))

(defun nl-llm-learn-copy-curriculum--example (index length literal)
  (list :index index :length length :literal literal
        :prompt (format "COPY: %s\nOUTPUT:\n" literal)
        :completion (concat literal "\n")))

;;;###autoload
(defun nl-llm-learn-copy-curriculum-dataset ()
  "Return the deterministic 160-example length-ordered curriculum.

Twenty unique strings are accepted at each configured length.  Rejected
duplicates still consume their generated characters and the PRNG state is
carried into the next length.  Global accepted indices divisible by five form
the 32-example development set; the other 128 examples are training data."
  (let ((state nl-llm-learn-copy-curriculum-seed)
        (seen (make-hash-table :test #'equal))
        (all nil) (train nil) (dev nil) (index 0))
    (dolist (length nl-llm-learn-copy-curriculum-lengths)
      (let ((accepted 0) (attempts 0))
        (while (< accepted nl-llm-learn-copy-curriculum-per-length)
          (setq attempts (1+ attempts))
          (when (> attempts 10000)
            (error "curriculum duplicate rejection exceeded bound at length %d"
                   length))
          (let ((chars nil))
            (dotimes (_ length)
              (setq state
                    (nl-llm-learn-copy-curriculum--xorshift32 state))
              (push (aref nl-llm-learn-copy-curriculum-alphabet
                          (mod state (length nl-llm-learn-copy-curriculum-alphabet)))
                    chars))
            (let ((literal (apply #'string (nreverse chars))))
              (unless (gethash literal seen)
                (puthash literal t seen)
                (let ((example
                       (nl-llm-learn-copy-curriculum--example
                        index length literal)))
                  (push example all)
                  (if (= (% index 5) 0)
                      (push example dev)
                    (push example train)))
                (setq accepted (1+ accepted)
                      index (1+ index))))))))
    (list :format nl-llm-learn-copy-curriculum-data-format
          :seed nl-llm-learn-copy-curriculum-seed
          :alphabet nl-llm-learn-copy-curriculum-alphabet
          :lengths (copy-sequence nl-llm-learn-copy-curriculum-lengths)
          :all (vconcat (nreverse all))
          :train (vconcat (nreverse train))
          :dev (vconcat (nreverse dev))
          :final-state state)))

(defun nl-llm-learn-copy-curriculum--examples (dataset key)
  (vconcat
   (mapcar (lambda (example)
             (list :prompt (plist-get example :prompt)
                   :completion (plist-get example :completion)))
           (append (plist-get dataset key) nil))))

(defun nl-llm-learn-copy-curriculum--sha256 (value)
  (let ((print-length nil) (print-level nil) (print-circle nil)
        (print-escape-nonascii t) (float-output-format nil))
    (secure-hash 'sha256 (prin1-to-string value))))

(defun nl-llm-learn-copy-curriculum--data-digest (dataset key)
  (nl-llm-learn-copy-curriculum--sha256
   (list :format nl-llm-learn-copy-curriculum-data-format
         :seed nl-llm-learn-copy-curriculum-seed
         :alphabet nl-llm-learn-copy-curriculum-alphabet
         :lengths nl-llm-learn-copy-curriculum-lengths
         :examples (append (plist-get dataset key) nil))))

;;;###autoload
(defun nl-llm-learn-copy-curriculum-train (model)
  "Train MODEL on the fixed 128-example curriculum training split.

This function does not create a model.  It owns GPU enable/free/disable and
returns bounded training metadata after one fixed 4096-step run."
  (when (nl-llm-gpu-available-p)
    (error "copy curriculum training refuses an already-active GPU"))
  (let* ((dataset (nl-llm-learn-copy-curriculum-dataset))
         (train-examples
          (nl-llm-learn-copy-curriculum--examples dataset :train))
         (encoded
          (nl-llm-agent-supervised-encode
           train-examples nl-llm-learn-copy-curriculum-tokenizer))
         (gpu-enabled nil)
         (context nil)
         (steps nil))
    (unwind-protect
        (progn
          (setq gpu-enabled (nl-llm-gpu-enable))
          (unless gpu-enabled
            (error "copy curriculum training requires a Vulkan GPU"))
          (setq context
                (nl-llm-agent-ondevice-from-model
                 model nl-llm-learn-copy-curriculum-sequence
                 nl-llm-learn-copy-curriculum-learning-rate
                 :optimizer 'adam :loss-mode 'completion
                 :transfer-mode 'compact))
          (setq steps
           (nl-llm-agent-ondevice-train
                 context (plist-get encoded :trajectories)
                 nl-llm-learn-copy-curriculum-epochs
                 :loss-starts (plist-get encoded :loss-starts)
                 :after-step
                 (lambda (_context completed _total)
                   (when (= (% completed 128) 0)
                     (message "COPY curriculum training: %d/4096 steps"
                              completed)))))
          (unless (= steps (* nl-llm-learn-copy-curriculum-epochs
                              (length (plist-get dataset :train))))
            (error "copy curriculum training completed %S steps, expected 4096"
                   steps))
          (nl-llm-agent-ondevice-sync context)
          (list :steps steps
                :train-count (length (plist-get dataset :train))
                :train-dataset-sha256 (plist-get encoded :dataset-sha256)
                :train-data-sha256
                (nl-llm-learn-copy-curriculum--data-digest dataset :train)))
      (when context
        (unwind-protect
            (nl-llm-agent-ondevice-free context)
          (when gpu-enabled
            (nl-llm-gpu-disable))))
      (when (and gpu-enabled (not context))
        (nl-llm-gpu-disable)))))

(defun nl-llm-learn-copy-curriculum--score-case (native example)
  "Score one EXAMPLE with fresh native caches and unrestricted byte output."
  (let* ((prompt-ids
          (nl-llm-agent-tokenizer-encode
           (plist-get example :prompt)
           nl-llm-learn-copy-curriculum-tokenizer))
         (expected
          (vconcat
           (nl-llm-agent-tokenizer-encode
            (plist-get example :completion)
            nl-llm-learn-copy-curriculum-tokenizer)))
         (blocks (plist-get native :blocks))
         (dim (plist-get native :dim))
         (heads (plist-get native :heads))
         (kvh (plist-get native :kvh))
         (caches
          (mapcar (lambda (_block)
                    (nl-llm-dcache-new
                     (+ (length prompt-ids)
                        nl-llm-learn-copy-curriculum-max-generated)
                     dim heads kvh))
                  blocks))
         (step
          (lambda (token)
            (nl-llm-decode-step
             token blocks caches
             (plist-get native :wte) (plist-get native :lnfg)
             (plist-get native :bh) dim nil (plist-get native :wh))))
         (decoded
          (nl-llm-learn-literal-copy-greedy-decode
           prompt-ids step nl-llm-learn-copy-curriculum-max-generated))
         (actual (plist-get decoded :ids))
         (terminated (plist-get decoded :terminated)))
    (list :index (plist-get example :index)
          :length (plist-get example :length)
          :literal (plist-get example :literal)
          :ids actual
          :expected-length (length expected)
          :generated-length (length actual)
          :terminated terminated
          :exact (and terminated (equal actual expected)))))

(defun nl-llm-learn-copy-curriculum--score (model examples label)
  "Return bounded exact-copy scores for MODEL and EXAMPLES under LABEL."
  (let* ((native (nl-llm-learn-literal-copy--native-model model))
         (groups nil)
         (passed 0)
         (total (length examples)))
    (dolist (length nl-llm-learn-copy-curriculum-lengths)
      (let ((members
             (cl-remove-if-not
              (lambda (example) (= (plist-get example :length) length))
              (append examples nil)))
            (cases nil) (group-passed 0) (failures nil))
        (dolist (example members)
          (let ((case (nl-llm-learn-copy-curriculum--score-case
                       native example)))
            (when (plist-get case :exact)
              (setq group-passed (1+ group-passed)
                    passed (1+ passed)))
            (if (eq label 'dev)
                (push case cases)
              (when (and (not (plist-get case :exact))
                         (< (length failures) 2))
                (push case failures)))))
        (when members
          (push (list :length length
                      :total (length members)
                      :passed group-passed
                      :score (/ (float group-passed) (length members))
                      (if (eq label 'dev) :cases :failures)
                      (vconcat (nreverse (if (eq label 'dev)
                                             cases failures))))
                groups))))
    (list :label label :passed passed :total total
          :score (if (> total 0) (/ (float passed) total) 0.0)
          :length-groups (vconcat (nreverse groups)))))

(defun nl-llm-learn-copy-curriculum--scores (model dataset)
  "Evaluate MODEL in byte-code mode on both fixed splits."
  (when (nl-llm-gpu-available-p)
    (error "copy curriculum scoring refuses an already-active GPU"))
  (unwind-protect
      (progn
        (nl-llm-inference-runtime-prepare 'byte-code)
        (list :train
              (nl-llm-learn-copy-curriculum--score
               model (plist-get dataset :train) 'train)
              :dev
              (nl-llm-learn-copy-curriculum--score
               model (plist-get dataset :dev) 'dev)))
    (nl-llm-inference-runtime-prepare 'source)))

;;;###autoload
(defun nl-llm-learn-copy-curriculum-run ()
  "Run the fixed COPY curriculum pretraining probe and return a report."
  (when (nl-llm-gpu-available-p)
    (error "copy curriculum run refuses an already-active GPU"))
  (let* ((dataset (nl-llm-learn-copy-curriculum-dataset))
         (model
          (nl-llm-agent-initialization-create
           :initializer 'xorshift32
           :seed nl-llm-learn-copy-curriculum-initializer-seed
           :dim nl-llm-learn-copy-curriculum-dim
           :ff nl-llm-learn-copy-curriculum-ff
           :vocab nl-llm-learn-copy-curriculum-vocab
           :nblocks nl-llm-learn-copy-curriculum-blocks
           :heads nl-llm-learn-copy-curriculum-heads
           :tokenizer nl-llm-learn-copy-curriculum-tokenizer))
         (identity model)
         (settings
          (list :alphabet nl-llm-learn-copy-curriculum-alphabet
                :lengths nl-llm-learn-copy-curriculum-lengths
                :per-length nl-llm-learn-copy-curriculum-per-length
                :seed nl-llm-learn-copy-curriculum-seed
                :initializer 'xorshift32
                :initializer-seed nl-llm-learn-copy-curriculum-initializer-seed
                :dim nl-llm-learn-copy-curriculum-dim
                :ff nl-llm-learn-copy-curriculum-ff
                :vocab nl-llm-learn-copy-curriculum-vocab
                :blocks nl-llm-learn-copy-curriculum-blocks
                :heads nl-llm-learn-copy-curriculum-heads
                :sequence nl-llm-learn-copy-curriculum-sequence
                :loss-mode 'completion
                :learning-rate nl-llm-learn-copy-curriculum-learning-rate
                :optimizer 'adam :transfer-mode 'compact
                :epochs nl-llm-learn-copy-curriculum-epochs
                :max-generated nl-llm-learn-copy-curriculum-max-generated
                :steps (* nl-llm-learn-copy-curriculum-epochs
                          (length (plist-get dataset :train)))))
         (before-hash (nl-llm-learn-literal-copy--model-hash model))
         (before (nl-llm-learn-copy-curriculum--scores model dataset))
         (before-eval-hash (nl-llm-learn-literal-copy--model-hash model)))
    (unless (equal before-hash before-eval-hash)
      (error "before evaluation mutated the model"))
    (let ((training (nl-llm-learn-copy-curriculum-train model)))
      (let* ((after-hash (nl-llm-learn-literal-copy--model-hash model))
             (after (nl-llm-learn-copy-curriculum--scores model dataset))
             (after-eval-hash
              (nl-llm-learn-literal-copy--model-hash model)))
        (unless (equal after-hash after-eval-hash)
          (error "after evaluation mutated the model"))
        (unless (eq model identity)
          (error "training replaced the original model object"))
        (list :format nl-llm-learn-copy-curriculum-format
              :source-sha256 nl-llm-learn-copy-curriculum-source-sha256
              :settings settings
              :data
              (list :format nl-llm-learn-copy-curriculum-data-format
                    :all-count (length (plist-get dataset :all))
                    :train-count (length (plist-get dataset :train))
                    :dev-count (length (plist-get dataset :dev))
                    :all-sha256
                    (nl-llm-learn-copy-curriculum--data-digest dataset :all)
                    :train-sha256
                    (nl-llm-learn-copy-curriculum--data-digest dataset :train)
                    :dev-sha256
                    (nl-llm-learn-copy-curriculum--data-digest dataset :dev)
                    :final-prng-state (plist-get dataset :final-state))
              :model-before-sha256 before-hash
              :model-after-sha256 after-hash
              :training training
              :before before
              :after after
              :steps (plist-get training :steps)
              :weights-changed (not (equal before-hash after-hash))
              :model-identity-preserved t)))))

(when (and noninteractive nl-llm-learn-copy-curriculum-auto-run)
  (let ((print-length nil) (print-level nil) (print-circle nil)
        (print-escape-nonascii t))
    (prin1 (nl-llm-learn-copy-curriculum-run))
    (terpri)))

(provide 'learn-copy-curriculum)
;;; learn-copy-curriculum.el ends here
