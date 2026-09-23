;;; agent-forward-parity-test.el --- GPU/CPU/native P5 forward parity -*- lexical-binding: t; -*-
;; Run after other GPU work is idle:
;;   emacs -Q --batch -l test/agent-forward-parity-test.el

(setq load-prefer-newer t)
(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'cl-lib)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-agent-artifact)
(require 'nl-llm-agent-improve)
(require 'nl-llm-agent-tokenizer)
(require 'nl-llm-decode)
(require 'nl-llm-gpu)
(require 'nl-llm-gpu-ag)

(defvar afp--fail 0)

(defconst afp--block-keys
  '(:ln1g :wq :bq :wk :bk :wv :bv :wo :bo
    :ln2g :wg :bg :wu :bu :wd :bd))

(defconst afp--gpu-tolerance 5.0e-4
  "Maximum absolute forward difference allowed for GPU float arithmetic.")

(defconst afp--native-tolerance 1.0e-5
  "Maximum difference allowed between the two host P5 implementations.")

(defun afp--check (name ok &optional detail)
  (princ
   (format "%-62s %s%s\n" name
           (if ok "PASS"
             (setq afp--fail (1+ afp--fail))
             "FAIL")
           (if detail (concat "  " detail) ""))))

(defun afp--finite-vector-p (values)
  "Return non-nil when every element of VALUES is finite numeric data."
  (let ((finite t))
    (dotimes (index (length values))
      (let ((value (aref values index)))
        (unless (and (numberp value)
                     (= value value)
                     (<= (abs value) 1.7976931348623157e+308))
          (setq finite nil))))
    finite))

(defun afp--assert-cpu-dispatch ()
  "Assert that every GPU-overridden tensor op currently points to its CPU cell."
  (unless (and (boundp 'photon-tensor-gpu--saved)
               photon-tensor-gpu--saved)
    (error "forward parity has no saved CPU tensor dispatch"))
  (dolist (pair photon-tensor-gpu--saved)
    (unless (eq (symbol-function (car pair)) (cdr pair))
      (error "forward parity tensor op is not on CPU dispatch: %S" (car pair))))
  t)

(defun afp--maxdiff (left right)
  "Return maximum absolute difference between equal-length vectors."
  (unless (= (length left) (length right))
    (error "forward parity vector lengths differ: %d and %d"
           (length left) (length right)))
  (unless (afp--finite-vector-p left)
    (error "forward parity left vector contains non-finite data"))
  (unless (afp--finite-vector-p right)
    (error "forward parity right vector contains non-finite data"))
  (let ((maximum 0.0))
    (dotimes (index (length left))
      (setq maximum
            (max maximum
                 (abs (- (aref left index) (aref right index))))))
    maximum))

(defun afp--last-row (values vocab)
  "Return a detached final VOCAB-wide row from VALUES."
  (let* ((base (- (length values) vocab))
         (row (make-vector vocab 0.0)))
    (dotimes (column vocab)
      (aset row column (aref values (+ base column))))
    row))

(defun afp--gpu-logits (model tokens)
  "Return every GPU batch logit row for MODEL and token vector TOKENS."
  (let* ((seq (length tokens))
         (dim (plist-get model :dim))
         (heads (plist-get model :heads))
         (builder (nlga-new)))
    (cl-labels
        ((parameter (pav)
           (nlga-param builder (pav-value pav)))
         (block (source)
           (let ((result nil))
             (dolist (key afp--block-keys)
               (setq result
                     (append result
                             (list key (parameter (plist-get source key))))))
             result)))
      (let* ((token-rt
              (nlga-const
               builder
               (photon-tensor
                (list seq 1)
                (apply #'vector
                       (mapcar #'float (append tokens nil))))))
             (wte (parameter (plist-get model :wte)))
             (blocks (mapcar #'block (plist-get model :blocks)))
             (lnfg (parameter (plist-get model :lnfg)))
             (wh (parameter (plist-get model :wh)))
             (bh (parameter (plist-get model :bh)))
             (tables (nl-llm-gpu-rope-tables seq (/ dim heads)))
             (cosr (nlga-const builder (car tables)))
             (sinr (nlga-const builder (cdr tables)))
             (spos (nlga-scalar builder 1.0))
             (sneg (nlga-scalar builder -1.0))
             (one (nlga-scalar builder 1.0))
             (logits
              (nlga-model-idx
               builder token-rt wte blocks lnfg wh bh heads heads
               cosr sinr spos sneg nil nil))
             (output (nlga-keep builder logits one)))
        (unwind-protect
            (progn
              ;; This graph has no loss seed, backward pass, or optimizer.
              (nlga-compile builder)
              (copy-sequence (nth output (nlga-step builder))))
          (nlga-free builder))))))

(defun afp--cpu-logits (model tokens)
  "Return every CPU autograd batch logit row for MODEL and TOKENS.
The test pins the global photon-tensor dispatch to its pure-elisp cells."
  (afp--assert-cpu-dispatch)
  (copy-sequence
   (photon-tensor-data
    (pav-value
     (nl-llm-agent--p5-forward model (append tokens nil))))))

(defun afp--native-logits (model tokens)
  "Return every sequential native logit row for exported MODEL and TOKENS."
  (afp--assert-cpu-dispatch)
  (let* ((checkpoint (nl-llm-agent-artifact-export-pav model))
         (exported
          (nl-llm-agent-artifact--checkpoint-model
           (append (list :format nl-llm-ckpt-format) checkpoint)
           "forward-parity"))
         (dim (plist-get exported :dim))
         (heads (plist-get exported :heads))
         (kvh (plist-get exported :kvh))
         (caches
          (mapcar
           (lambda (_block)
             (nl-llm-dcache-new (length tokens) dim heads kvh))
           (plist-get exported :blocks)))
         (rows nil))
    (dotimes (index (length tokens))
      (push
       (nl-llm-decode-step
        (aref tokens index) (plist-get exported :blocks) caches
        (plist-get exported :wte) (plist-get exported :lnfg)
        (plist-get exported :bh) dim nil (plist-get exported :wh))
       rows))
    (apply #'vconcat (nreverse rows))))

(defun afp--perturb (model)
  "Apply deterministic nonzero perturbations to MODEL and return it."
  (cl-labels
      ((add (pav index delta)
         (let ((data (photon-tensor-data (pav-value pav))))
           (aset data index (+ (aref data index) delta)))))
    ;; Exercise embedding, block biases/norm, and especially the independent
    ;; output head plus nonzero head bias through export and native decoding.
    (add (plist-get model :wte) (+ (* 65 (plist-get model :dim)) 3) 0.31)
    (let ((block (car (plist-get model :blocks))))
      (add (plist-get block :ln1g) 2 -0.17)
      (add (plist-get block :bq) 1 0.23)
      (add (plist-get block :bk) 4 -0.19)
      (add (plist-get block :bv) 5 0.13)
      (add (plist-get block :bo) 6 -0.29)
      (add (plist-get block :bg) 3 0.11)
      (add (plist-get block :bd) 7 0.27))
    (add (plist-get model :lnfg) 5 0.21)
    (add (plist-get model :wh) (+ (* 230 (plist-get model :dim)) 1) 0.37)
    (add (plist-get model :bh) 0 -0.33)
    (add (plist-get model :bh) 230 0.41))
  model)

(defun afp--run-case (variant model label text)
  "Compare all three forward paths for VARIANT MODEL on UTF-8 TEXT."
  (let* ((tokens
          (apply #'vector
                 (nl-llm-agent-tokenizer-encode text "utf8-byte-v1")))
         (vocab (plist-get model :vocab))
         (gpu (afp--gpu-logits model tokens))
         (cpu (afp--cpu-logits model tokens))
         (native (afp--native-logits model tokens))
         (gpu-cpu (afp--maxdiff gpu cpu))
         (cpu-native
          (afp--maxdiff (afp--last-row cpu vocab)
                        (afp--last-row native vocab)))
         (gpu-native
          (afp--maxdiff (afp--last-row gpu vocab)
                        (afp--last-row native vocab))))
    (afp--check
     (format "%s %s: every GPU row matches CPU batch" variant label)
     (< gpu-cpu afp--gpu-tolerance)
     (format "maxdiff=%.3g" gpu-cpu))
    (afp--check
     (format "%s %s: final CPU row matches exported native" variant label)
     (< cpu-native afp--native-tolerance)
     (format "maxdiff=%.3g" cpu-native))
    (afp--check
     (format "%s %s: final GPU row matches exported native" variant label)
     (< gpu-native afp--gpu-tolerance)
     (format "maxdiff=%.3g" gpu-native))))

(unless (nl-llm-gpu-enable)
  (princ "agent forward parity: SKIP (no Vulkan device)\n")
  (kill-emacs 0))

;; Keep host CPU comparisons on the saved pure-elisp cells.  The direct NLGA
;; graph above bypasses these function cells and remains GPU-backed.
(photon-tensor-use-cpu-backend)
(afp--assert-cpu-dispatch)

(unwind-protect
    (let ((initial (nl-llm-agent-improve-model 8 12 256 1 2 "utf8-byte-v1"))
          (perturbed
           (afp--perturb
            (nl-llm-agent-improve-model 8 12 256 1 2 "utf8-byte-v1")))
          ;; These include multiple prefix lengths, a repeated ASCII token, and
          ;; repeated multibyte token subsequences including a supplementary
          ;; Unicode scalar.
          (cases '(("short-repeat" . "AA")
                   ("mixed-prefix" . "A日A")
                   ("supplementary-repeat" . "🙂🙂"))))
      (dolist (case cases)
        (afp--run-case "initial" initial (car case) (cdr case)))
      (dolist (case cases)
        (afp--run-case "perturbed" perturbed (car case) (cdr case))))
  (nl-llm-gpu-disable))

(when (> afp--fail 0)
  (error "agent forward parity: %d failure(s)" afp--fail))
(princ "agent forward parity: all checks passed\n")

;;; agent-forward-parity-test.el ends here
