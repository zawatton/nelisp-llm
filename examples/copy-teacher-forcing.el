;;; copy-teacher-forcing.el --- teacher-forced COPY diagnostics -*- lexical-binding: t; -*-

;; This is a CPU-only diagnostic for separating per-token prediction from the
;; frozen literal-copy probe's exact free-running score.  It never trains,
;; samples, or runs automatically when loaded.

;;; Code:

(require 'cl-lib)

(defvar nl-llm-learn-copy-curriculum-auto-run nil)
(defvar nl-llm-learn-literal-copy-no-run nil)

;; Loading the curriculum also loads the frozen native export/decoder helpers.
;; Bind both guards before loading so an ambient example setting cannot start a
;; GPU run as a side effect of requiring this diagnostic.
(let* ((here (file-name-directory (or load-file-name buffer-file-name)))
       (curriculum (expand-file-name "learn-copy-curriculum.el" here)))
  (unless (featurep 'learn-copy-curriculum)
    (let ((nl-llm-learn-copy-curriculum-auto-run nil)
          (nl-llm-learn-literal-copy-no-run t))
      (load curriculum nil nil t))))

(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'nl-llm-agent-tokenizer)
(require 'nl-llm-decode)
(require 'nl-llm-gpu)
(require 'nl-llm-inference-runtime)

(defconst nl-llm-copy-teacher-forcing-format
  "nl-llm-copy-teacher-forcing-report-v1")
(defconst nl-llm-copy-teacher-forcing-max-examples 128)
(defconst nl-llm-copy-teacher-forcing-max-tokens 64)
(defconst nl-llm-copy-teacher-forcing-max-chars 4096)

(declare-function nl-llm-agent-tokenizer-id
                  "nl-llm-agent-tokenizer" (&optional identifier))
(declare-function nl-llm-agent-tokenizer-encode
                  "nl-llm-agent-tokenizer" (text &optional identifier))
(declare-function nl-llm-agent-tokenizer-vocab
                  "nl-llm-agent-tokenizer" (&optional identifier))
(declare-function nl-llm-gpu-available-p "nl-llm-gpu" ())
(declare-function nl-llm-inference-runtime-prepare
                  "nl-llm-inference-runtime" (&optional mode))
(declare-function nl-llm-dcache-new "nl-llm-decode"
                  (max-seq dim heads kvh))
(declare-function nl-llm-decode-step "nl-llm-decode"
                  (token blocks caches wte lnfg bh dim &optional rope-base head))
(declare-function nl-llm-learn-literal-copy--native-model
                  "learn-literal-copy" (model))
(declare-function nl-llm-learn-literal-copy--model-hash
                  "learn-literal-copy" (model))

(defun nl-llm-copy-teacher-forcing--keys (value where)
  "Validate the supported example plist VALUE for WHERE." 
  (unless (listp value)
    (error "%s must be a plist" where))
  (let ((tail value) (seen nil))
    (while tail
      (unless (and (consp tail) (consp (cdr tail)))
        (error "%s must be a proper plist" where))
      (let ((key (car tail)))
        (unless (memq key '(:prompt :completion :index :length :literal))
          (error "%s has unknown key %S" where key))
        (when (memq key seen)
          (error "%s has duplicate key %S" where key))
        (push key seen))
      (setq tail (cddr tail)))
    (unless (and (memq :prompt seen) (memq :completion seen))
      (error "%s requires :prompt and :completion" where)))
  value)

(defun nl-llm-copy-teacher-forcing--text (value where)
  "Return detached bounded non-empty TEXT for WHERE." 
  (unless (and (stringp value)
               (> (length value) 0)
               (<= (length value) nl-llm-copy-teacher-forcing-max-chars))
    (error "%s must be non-empty text of at most %d characters"
           where nl-llm-copy-teacher-forcing-max-chars))
  (substring-no-properties value))

(defun nl-llm-copy-teacher-forcing--finite-p (value)
  "Return non-nil for a finite numeric VALUE without relying on `isinf'." 
  (and (numberp value)
       (= value value)
       (< (abs value) 1.7976931348623157e+308)))

(defun nl-llm-copy-teacher-forcing--prepare-examples (examples tokenizer)
  "Validate and encode EXAMPLES for TOKENIZER, returning detached records." 
  (unless (and (vectorp examples)
               (> (length examples) 0)
               (<= (length examples) nl-llm-copy-teacher-forcing-max-examples))
    (error "teacher-forcing examples must be a vector containing 1..%d entries"
           nl-llm-copy-teacher-forcing-max-examples))
  (let ((records nil))
    (dotimes (index (length examples))
      (let* ((example (aref examples index))
             (_keys (nl-llm-copy-teacher-forcing--keys
                     example (format "teacher-forcing example %d" index)))
             (prompt (nl-llm-copy-teacher-forcing--text
                      (plist-get example :prompt)
                      (format "teacher-forcing example %d prompt" index)))
             (completion (nl-llm-copy-teacher-forcing--text
                          (plist-get example :completion)
                          (format "teacher-forcing example %d completion" index)))
             (prefix (substring completion 0 (1- (length completion))))
             (newline-count (cl-count ?\n completion)))
        ;; The diagnostic is specifically for literal-copy records, whose
        ;; completion has exactly one terminal newline and no earlier newline.
        (unless (and (= (aref completion (1- (length completion))) ?\n)
                     (= newline-count 1))
          (error "teacher-forcing example %d completion must end in exactly one newline"
                 index))
        (let* ((prompt-ids (nl-llm-agent-tokenizer-encode prompt tokenizer))
               (completion-ids
                (nl-llm-agent-tokenizer-encode completion tokenizer))
               (tokens (+ (length prompt-ids) (length completion-ids))))
          (unless (and (> (length prompt-ids) 0)
                       (> (length completion-ids) 0)
                       (<= tokens nl-llm-copy-teacher-forcing-max-tokens))
            (error "teacher-forcing example %d encoded length must be 1..%d"
                   index nl-llm-copy-teacher-forcing-max-tokens))
          (push (list :index (if (integerp (plist-get example :index))
                                 (plist-get example :index) index)
                      :length (if (integerp (plist-get example :length))
                                  (plist-get example :length)
                                (length prefix))
                      :prompt-ids (copy-sequence prompt-ids)
                      :completion-ids (vconcat completion-ids))
                records))))
    (nreverse records)))

(defun nl-llm-copy-teacher-forcing--logits (logits target vocab where)
  "Return (CORRECT LOSS ARGMAX) for LOGITS and TARGET in WHERE." 
  (unless (and (vectorp logits) (= (length logits) vocab))
    (error "%s logits must be a vector of length %d" where vocab))
  (let ((maximum -1.7976931348623157e+308)
        (argmax 0)
        (sum 0.0)
        (target-value nil))
    (dotimes (id vocab)
      (let ((value (aref logits id)))
        (unless (nl-llm-copy-teacher-forcing--finite-p value)
          (error "%s logits contain a non-finite value at %d" where id))
        (when (> value maximum)
          (setq maximum value argmax id))))
    (unless (and (integerp target) (<= 0 target) (< target vocab))
      (error "%s target is outside vocabulary: %S" where target))
    (setq target-value (aref logits target))
    (dotimes (id vocab)
      (setq sum (+ sum (exp (- (aref logits id) maximum)))))
    ;; Keep the log-sum-exp translation invariant even when MAXIMUM is large.
    (let ((loss (+ (- maximum target-value) (log sum))))
      (unless (nl-llm-copy-teacher-forcing--finite-p loss)
        (error "%s cross-entropy is non-finite" where))
      (list (= argmax target) loss argmax))))

(defun nl-llm-copy-teacher-forcing--case (native record)
  "Score RECORD by feeding gold completion prefixes to NATIVE." 
  (let* ((prompt-ids (plist-get record :prompt-ids))
         (expected (plist-get record :completion-ids))
         (blocks (plist-get native :blocks))
         (dim (plist-get native :dim))
         (heads (plist-get native :heads))
         (kvh (plist-get native :kvh))
         (caches (mapcar (lambda (_block)
                           (nl-llm-dcache-new
                            (+ (length prompt-ids) (length expected))
                            dim heads kvh))
                         blocks))
         (step (lambda (token)
                 (nl-llm-decode-step
                  token blocks caches
                  (plist-get native :wte) (plist-get native :lnfg)
                  (plist-get native :bh) dim nil (plist-get native :wh))))
         (logits nil)
         (correct 0)
         (loss 0.0)
         (first-correct 0)
         (newline-correct 0)
         (prompt-tail prompt-ids)
         (index 0)
         ;; Artifact checkpoints keep vocabulary in their validated config;
         ;; unlike the trainable model, the returned inference plist has no
         ;; top-level :vocab field.
         (vocab (plist-get (plist-get native :config) :vocab)))
    ;; Prefill only the prompt.  The first target is scored from the final
    ;; prompt logit, then each subsequent input is the preceding GOLD token.
    (while prompt-tail
      (setq logits (funcall step (car prompt-tail))
            prompt-tail (cdr prompt-tail)))
    (while (< index (length expected))
      (let* ((target (aref expected index))
             (stats
              (nl-llm-copy-teacher-forcing--logits
               logits target vocab
               (format "teacher-forcing case %S token %d"
                       (plist-get record :index) index))))
        (when (car stats)
          (setq correct (1+ correct)))
        (when (= index 0)
          (setq first-correct (if (car stats) 1 0)))
        (when (= index (1- (length expected)))
          (setq newline-correct (if (car stats) 1 0)))
        (setq loss (+ loss (cadr stats)))
        ;; Do not feed the final target: there is no next target to score.
        (when (< (1+ index) (length expected))
          (setq logits (funcall step target)))
        (setq index (1+ index))))
    (list :index (plist-get record :index)
          :length (plist-get record :length)
          :tokens (length expected)
          :correct correct
          :accuracy (/ (float correct) (length expected))
          :first-byte-correct first-correct
          :newline-correct newline-correct
          :loss loss
          :mean-loss (/ loss (float (length expected))))))

(defun nl-llm-copy-teacher-forcing--sum (cases length)
  "Aggregate CASES, optionally retaining only cases of LENGTH." 
  (let ((selected (if length
                      (cl-remove-if-not
                       (lambda (case) (= (plist-get case :length) length))
                       cases)
                    cases))
        (tokens 0) (correct 0) (first 0) (newline 0) (loss 0.0))
    (dolist (case selected)
      (setq tokens (+ tokens (plist-get case :tokens))
            correct (+ correct (plist-get case :correct))
            first (+ first (plist-get case :first-byte-correct))
            newline (+ newline (plist-get case :newline-correct))
            loss (+ loss (plist-get case :loss))))
    (list :length length :cases (length selected) :tokens tokens
          :correct correct
          :accuracy (if (> tokens 0) (/ (float correct) tokens) 0.0)
          :first-byte-correct first :newline-correct newline
          :loss loss :mean-loss (if (> tokens 0) (/ loss (float tokens)) 0.0))))

;;;###autoload
(defun nl-llm-copy-teacher-forcing-score (model examples)
  "Return a CPU teacher-forced next-byte diagnostic for MODEL and EXAMPLES.

EXAMPLES is a vector of COPY records with non-empty :prompt and a completion
ending in exactly one newline.  Every target is scored from the prompt and the
preceding gold completion bytes; no generated or future completion byte is
fed before its score.  The result is a diagnostic, not a free-running score or
a competence claim.  MODEL and its weights are never mutated." 
  (when (nl-llm-gpu-available-p)
    (error "teacher-forcing scoring refuses an already-active GPU"))
  (let* ((native (nl-llm-learn-literal-copy--native-model model))
         (tokenizer (nl-llm-agent-tokenizer-id (plist-get native :tokenizer)))
         (records (nl-llm-copy-teacher-forcing--prepare-examples
                   examples tokenizer))
         (before (nl-llm-learn-literal-copy--model-hash model))
         (cases nil))
    (unless (equal tokenizer "utf8-byte-v1")
      (error "teacher-forcing COPY diagnostic requires utf8-byte-v1"))
    (unwind-protect
        (progn
            (nl-llm-inference-runtime-prepare 'byte-code)
            (dolist (record records)
            (push (nl-llm-copy-teacher-forcing--case
                   native record)
                  cases))
          (setq cases (nreverse cases)))
      (nl-llm-inference-runtime-prepare 'source))
    (let* ((after (nl-llm-learn-literal-copy--model-hash model))
           (total (nl-llm-copy-teacher-forcing--sum cases nil))
           (lengths (delete-dups
                     (mapcar (lambda (case) (plist-get case :length)) cases)))
           (groups
            (mapcar (lambda (length)
                      (nl-llm-copy-teacher-forcing--sum cases length))
                    (sort lengths #'<))))
      (unless (equal before after)
        (error "teacher-forcing evaluation mutated model weights"))
      (list :format nl-llm-copy-teacher-forcing-format
            :mode 'teacher-forced
            :tokenizer (copy-sequence tokenizer)
            :cases (vconcat cases)
            :per-length (vconcat groups)
            :total (plist-put total :length nil)
            :model-unchanged t))))

(provide 'copy-teacher-forcing)
;;; copy-teacher-forcing.el ends here
