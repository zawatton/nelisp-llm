;;; agent-gradient-parity-test.el --- full P5 GPU/CPU gradient parity -*- lexical-binding: t; -*-
;; Run after other GPU work is idle:
;;   emacs -Q --batch -l test/agent-gradient-parity-test.el

(setq load-prefer-newer t)
(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'cl-lib)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-agent-improve)
(require 'nl-llm-agent-tokenizer)
(require 'nl-llm-gpu)
(require 'nl-llm-gpu-ag)

(defvar agp--fail 0)

(defconst agp--block-keys
  '(:ln1g :wq :bq :wk :bk :wv :bv :wo :bo
    :ln2g :wg :bg :wu :bu :wd :bd))

(defun agp--check (name ok &optional detail)
  (princ
   (format "%-68s %s%s\n" name
           (if ok "PASS"
             (setq agp--fail (1+ agp--fail))
             "FAIL")
           (if detail (concat "  " detail) ""))))

(defun agp--finite-vector-p (values)
  "Return non-nil when every element of VALUES is finite numeric data."
  (let ((finite t))
    (dotimes (index (length values))
      (let ((value (aref values index)))
        (unless (and (numberp value)
                     (= value value)
                     (< (abs value) 1.7976931348623157e+308))
          (setq finite nil))))
    finite))

(defun agp--assert-cpu-dispatch ()
  "Assert that tensor function cells currently point at their pure CPU cells."
  (unless (and (boundp 'photon-tensor-gpu--saved)
               photon-tensor-gpu--saved)
    (error "gradient parity has no saved CPU tensor dispatch"))
  (dolist (pair photon-tensor-gpu--saved)
    (unless (eq (symbol-function (car pair)) (cdr pair))
      (error "gradient parity tensor op is not on CPU dispatch: %S" (car pair))))
  t)

(defun agp--named-params (model)
  "Return MODEL parameters in the canonical P5 order, with diagnostic names."
  (append
   (list (cons "wte" (plist-get model :wte)))
   (cl-loop for bi from 0
            for block in (plist-get model :blocks)
            append
            (mapcar (lambda (key)
                      (cons (format "block%d/%s" bi (substring (symbol-name key) 1))
                            (plist-get block key)))
                    agp--block-keys))
   (list (cons "lnfg" (plist-get model :lnfg))
         (cons "wh" (plist-get model :wh))
         (cons "bh" (plist-get model :bh)))))

(defun agp--parameter-values (model)
  "Return detached parameter arrays in canonical P5 order."
  (mapcar (lambda (entry)
            (copy-sequence
             (photon-tensor-data (pav-value (cdr entry)))))
          (agp--named-params model)))

(defun agp--parameter-hash (values)
  "Return a stable hash for detached parameter VALUES."
  (sxhash-equal values))

(defun agp--perturb (model)
  "Apply deterministic nonzero perturbations to MODEL and return it."
  (cl-labels
      ((add (pav index delta)
         (let ((data (photon-tensor-data (pav-value pav))))
           (aset data index (+ (aref data index) delta)))))
    ;; Exercise embedding, norm, attention/MLP biases, and the independent head.
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

(defun agp--case-data (text completion-start pad-token)
  "Make a fixed-length 16-row input/target/mask case from TEXT.
COMPLETION-START is a zero-based token index greater than zero in the
unpadded token stream; the target row immediately before that token is the
first active row.  CPU reference fields deliberately omit padding."
  (let* ((seq 16)
         (real (append
                (nl-llm-agent-tokenizer-encode text "utf8-byte-v1") nil))
         (real-length (length real)))
    (unless (and (>= real-length 2) (<= real-length (1+ seq)))
      (error "gradient parity case has invalid token length: %S" real))
    (unless (and (integerp completion-start)
                 (> completion-start 0)
                 (< completion-start real-length))
      (error "gradient parity case has invalid completion boundary: %S"
             completion-start))
    (let* ((full (append real (make-list (- (1+ seq) real-length) pad-token)))
           (inputs (vconcat (cl-subseq full 0 seq)))
           (targets (vconcat (cl-subseq full 1 (1+ seq))))
           (mask (make-vector seq 0))
           (cpu-inputs (butlast real))
           (cpu-targets (vconcat (cdr real)))
           (cpu-mask (make-vector (1- real-length) 0)))
      (dotimes (row seq)
        (when (and (>= row (1- completion-start))
                   (< row (1- real-length)))
          (aset mask row 1)))
      (dotimes (row (1- real-length))
        (when (>= row (1- completion-start))
          (aset cpu-mask row 1)))
      (list :label text :real real :inputs inputs :targets targets :mask mask
            :cpu-inputs cpu-inputs :cpu-targets cpu-targets :cpu-mask cpu-mask
            :completion-start completion-start :pad pad-token))))

(defun agp--gpu-gradients (model case)
  "Return (FIRST SECOND), each a list of GPU parameter gradients.
FIRST and SECOND are produced by two submissions of one compiled graph; the
second run detects accidental accumulation in temporary/gradient slots."
  (let* ((seq 16)
         (dim (plist-get model :dim))
         (heads (plist-get model :heads))
         (entries (agp--named-params model))
         (builder (nlga-new)))
    (cl-labels
        ((parameter (pav)
           (nlga-param builder (pav-value pav)))
         (block (source)
           (let ((result nil))
             (dolist (key agp--block-keys)
               (setq result
                     (append result
                             (list key (parameter (plist-get source key))))))
             result)))
      (let* ((input-rt
              (nlga-const
               builder
               (photon-tensor
                (list seq 1)
                (apply #'vector
                       (mapcar #'float (append (plist-get case :inputs) nil))))))
             (target-rt
              (nlga-const
               builder
               (photon-tensor
                (list seq 1)
                (apply #'vector
                       (mapcar #'float (append (plist-get case :targets) nil))))))
             (scale-rt
              (nlga-const
               builder
               (photon-tensor
                (list seq 1)
                (let ((active (cl-count 1 (plist-get case :mask))))
                  (unless (> active 0)
                    (error "gradient parity case has no active rows"))
                  (apply #'vector
                         (mapcar (lambda (x)
                                   (if (= x 1) (/ (float seq) active) 0.0))
                                 (append (plist-get case :mask) nil)))))))
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
             (logits
              (nlga-model-idx
               builder input-rt wte blocks lnfg wh bh heads heads
               cosr sinr spos sneg nil nil)))
        ;; Emit the complete backward graph, but deliberately no optimizer.
        (nlga-seed-ce-idx-masked builder logits target-rt scale-rt)
        (dolist (th (nlga-bwd builder)) (funcall th))
        (let ((one (nlga-scalar builder 1.0))
              (outputs nil))
          (dolist (entry entries)
            (let* ((pav (cdr entry))
                   (rt
                    (cl-find-if
                     (lambda (candidate)
                       (eq (plist-get candidate :tensor) (pav-value pav)))
                     (nlga-params builder)))
                   (resident (and rt (plist-get rt :rt)))
                   (grad-slot (and resident (nlga-rt-grad resident))))
              (unless (and resident grad-slot)
                (error "gradient parity missing GPU grad slot for %s" (car entry)))
              (push
               (nlga-keep
                builder
                (nlga-rt--make
                 :slot grad-slot :rows (nlga-rt-rows resident)
                 :cols (nlga-rt-cols resident))
                one)
               outputs)))
          (setq outputs (nreverse outputs))
          (unwind-protect
              (progn
                (nlga-compile builder)
                (let ((first (nlga-step builder))
                      (second nil))
                  (setq second (nlga-step builder))
                  (list
                   (mapcar #'copy-sequence (mapcar (lambda (index) (nth index first)) outputs))
                   (mapcar #'copy-sequence (mapcar (lambda (index) (nth index second)) outputs)))))
            (nlga-free builder)))))))

(defun agp--cpu-gradients (model case)
  "Return detached CPU autograd gradients for CASE, with CPU dispatch pinned."
  (agp--assert-cpu-dispatch)
  (let* ((entries (agp--named-params model))
         (params (mapcar #'cdr entries))
         (loss
          (nl-llm-agent--p5-forward
           model
           (copy-sequence (plist-get case :cpu-inputs))
           (copy-sequence (plist-get case :cpu-targets))
           (copy-sequence (plist-get case :cpu-mask)))))
    (photon-autograd-zero-grad params)
    (photon-autograd-backward loss)
    (mapcar (lambda (p)
              (let ((grad (pav-grad p)))
                (unless grad (error "CPU gradient is absent"))
                (copy-sequence (photon-tensor-data grad))))
            params)))

(defun agp--max-error (reference actual)
  "Return (MAX-DIFF REF-MAX WORST-INDEX MAX-VIOLATION) for finite vectors."
  (unless (= (length reference) (length actual))
    (error "gradient parity vector lengths differ: %d and %d"
           (length reference) (length actual)))
  (unless (and (agp--finite-vector-p reference)
               (agp--finite-vector-p actual))
    (error "gradient parity encountered a non-finite gradient"))
  (let ((maximum 0.0) (refmax 0.0) (worst 0) (max-violation 0.0))
    (dotimes (i (length reference))
      (let ((err (abs (- (aref actual i) (aref reference i)))))
        (when (> err maximum) (setq maximum err worst i))
        (setq refmax (max refmax (abs (aref reference i))))
        (setq max-violation
              (max max-violation
                   (- err (+ 2.0e-5 (* 2.0e-4 (abs (aref reference i)))))))))
    (list maximum refmax worst max-violation)))

(defun agp--max-nonzero (values &optional threshold)
  "Return (INDEX ABS-VALUE) for the largest nonzero element of VALUES."
  (let ((threshold (or threshold 1.0e-10)) (index nil) (maximum 0.0))
    (dotimes (i (length values))
      (let ((magnitude (abs (aref values i))))
        (when (and (> magnitude maximum) (> magnitude threshold))
          (setq index i maximum magnitude))))
    (and index (list index maximum))))

(defun agp--cpu-loss (model case)
  "Evaluate CASE loss on the pure CPU dispatch."
  (agp--assert-cpu-dispatch)
  (let ((loss
         (nl-llm-agent--p5-forward
          model
          (copy-sequence (plist-get case :cpu-inputs))
          (copy-sequence (plist-get case :cpu-targets))
          (copy-sequence (plist-get case :cpu-mask)))))
    (aref (photon-tensor-data (pav-value loss)) 0)))

(defun agp--finite-difference (model case analytic)
  "Check one selected central difference coordinate in every MODEL parameter."
  (dolist (entry (cl-mapcar #'cons (agp--named-params model) analytic))
    (let* ((name (car (car entry)))
           (pav (cdr (car entry)))
           (gradient (cdr entry))
           (selected (agp--max-nonzero gradient))
           (data (photon-tensor-data (pav-value pav))))
      (if (not selected)
          (agp--check (format "CPU finite difference %s (zero recorded)" name)
                      t "no coordinate above 1e-10")
        (let* ((index (car selected))
               (reference (aref gradient index))
               (original (aref data index))
               (epsilon 1.0e-4)
               plus minus numerical)
          (unwind-protect
              (progn
                (aset data index (+ original epsilon))
                (setq plus (agp--cpu-loss model case))
                (aset data index (- original epsilon))
                (setq minus (agp--cpu-loss model case))
                (setq numerical (/ (- plus minus) (* 2.0 epsilon)))
                (agp--check
                 (format "CPU finite difference %s[%d]" name index)
                 (and (= numerical numerical)
                      (<= (abs (- numerical reference))
                          (+ 1.0e-5 (* 5.0e-4 (abs reference)))))
                 (format "analytic=%.6g numeric=%.6g" reference numerical)))
            (aset data index original)))))))

(defun agp--run-case (variant model case seen)
  "Compare GPU and CPU full gradients for one padded CASE.
SEEN is a hash table recording parameters that obtained nonzero CPU gradients."
  (let* ((gpu-pair (agp--gpu-gradients model case))
         (gpu (car gpu-pair))
         (repeat (cadr gpu-pair))
         (cpu (agp--cpu-gradients model case))
         (entries (agp--named-params model)))
    (cl-loop for entry in entries
             for gpu-gradient in gpu
             for repeated-gradient in repeat
             for cpu-gradient in cpu
             do
             (let* ((name (car entry))
                    (error-data (agp--max-error cpu-gradient gpu-gradient))
                    (repeat-data (agp--max-error gpu-gradient repeated-gradient))
                    (maxdiff (nth 0 error-data))
                    (refmax (nth 1 error-data))
                    (worst (nth 2 error-data))
                    (violation (nth 3 error-data))
                    (repeat-diff (nth 0 repeat-data)))
               (when (agp--max-nonzero cpu-gradient)
                 (puthash name t seen))
               (agp--check
                (format "%s %s: GPU/CPU gradient %s" variant
                        (plist-get case :label) name)
                (<= violation 0.0)
                (format "maxdiff=%.3g refmax=%.3g worst=%d" maxdiff refmax worst))
               (agp--check
                (format "%s %s: repeated GPU gradient %s" variant
                        (plist-get case :label) name)
                (<= repeat-diff 1.0e-7)
                (format "maxdiff=%.3g" repeat-diff))))
    cpu))

(unless (nl-llm-gpu-enable)
  (princ "agent gradient parity: SKIP (no Vulkan device)\n")
  (kill-emacs 0))

;; Keep host CPU comparisons on the saved pure-elisp cells.  Direct NLGA graph
;; dispatch remains GPU-backed while this global function-cell switch is active.
(photon-tensor-use-cpu-backend)
(agp--assert-cpu-dispatch)

(unwind-protect
    (let* ((initial (nl-llm-agent-improve-model 8 12 256 1 2 "utf8-byte-v1"))
           (perturbed
            (agp--perturb
             (nl-llm-agent-improve-model 8 12 256 1 2 "utf8-byte-v1")))
           ;; The fixed 16-row window has real and padded target rows.  Prompt
           ;; boundaries intentionally vary across ASCII and UTF-8 trajectories.
           (cases
            (list
             (agp--case-data "ABABAB" 2 0)
             (agp--case-data "ABAB\nAB" 5 7)
             (agp--case-data "A日A日A" 2 0)
             (agp--case-data "🙂日🙂A" 4 255)))
           (seen (make-hash-table :test #'equal))
           (initial-values (agp--parameter-values initial))
           (perturbed-values (agp--parameter-values perturbed))
           (initial-hash (agp--parameter-hash initial-values))
           (perturbed-hash (agp--parameter-hash perturbed-values))
           first-cpu)
      (dolist (case cases)
        (let ((cpu (agp--run-case "initial" initial case seen)))
          (unless first-cpu (setq first-cpu (cons case cpu)))))
      (dolist (case cases)
        (agp--run-case "perturbed" perturbed case seen))

      ;; Finite differences are pure CPU checks and cover one selected
      ;; nonzero coordinate in every parameter tensor where one exists.
      (agp--finite-difference initial (car first-cpu) (cdr first-cpu))

      (dolist (required '("wte" "block0/wq" "block0/wk" "wh"))
        (agp--check (format "nonzero attention/input gradient observed: %s" required)
                    (gethash required seen)))

      ;; No graph includes an optimizer, so all host values must remain exactly
      ;; unchanged after every GPU and finite-difference check.
      (let ((after-initial (agp--parameter-values initial))
            (after-perturbed (agp--parameter-values perturbed)))
        (agp--check "initial model weight values unchanged"
                    (equal initial-values after-initial)
                    (format "hash=%s" (agp--parameter-hash after-initial)))
        (agp--check "initial model weight hash unchanged"
                    (= initial-hash (agp--parameter-hash after-initial)))
        (agp--check "perturbed model weight values unchanged"
                    (equal perturbed-values after-perturbed)
                    (format "hash=%s" (agp--parameter-hash after-perturbed)))
        (agp--check "perturbed model weight hash unchanged"
                    (= perturbed-hash (agp--parameter-hash after-perturbed)))))
  (nl-llm-gpu-disable))

(when (> agp--fail 0)
  (error "agent gradient parity: %d failure(s)" agp--fail))
(princ "agent gradient parity: all checks passed\n")

;;; agent-gradient-parity-test.el ends here
