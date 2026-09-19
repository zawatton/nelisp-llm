;;; agent-recur-runtime-test.el --- recurrent CPU runtime adapter checks -*- lexical-binding: t; -*-

;; Run from this directory with:
;;   emacs -Q --batch -l test/agent-recur-runtime-test.el
;;
;; This is deliberately CPU-only.  It checks the shared in-memory byte-code
;; adapter against the source recurrent path without enabling a tensor backend.

;;; Code:

(let ((here (file-name-directory (or load-file-name buffer-file-name))))
  (add-to-list 'load-path (expand-file-name "../lisp" here))
  (add-to-list 'load-path (expand-file-name "../../nelisp-photon/lisp" here)))

(require 'cl-lib)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-autograd)
(require 'nl-llm-recur)
(require 'nl-llm-inference-runtime)
(require 'nl-llm-ckpt)
(require 'nl-llm-agent-artifact)
(require 'nl-llm-agent-improve)
(require 'nl-llm-agent-recur-provider)

(defvar arrt--fail 0)

(defun arrt--check (name ok &optional detail)
  (princ (format "%-70s %s  %s\n"
                 name
                 (if ok "PASS"
                   (progn (setq arrt--fail (1+ arrt--fail)) "FAIL"))
                 (or detail ""))))

(defun arrt--model ()
  "Return one deterministic, small recurrent GQA model."
  (nl-llm-recur-model-new
   :vocab 256 :dim 4 :heads 2 :kv-heads 1 :ff 8
   :n-prelude 1 :n-core 1 :n-coda 1 :seed 37 :sigma 0.35))

(defun arrt--dense-model ()
  "Return a tiny trainable P5 model for shared-runtime interleaving."
  ;; Keep this on the same trainable P5 representation used by the actual
  ;; forward path; artifact conversion is covered by the artifact tests.
  (nl-llm-agent-improve-model 2 2 256 1 1 "utf8-byte-v1"))

(defun arrt--dense-logits (model tokens)
  "Return detached dense full-sequence logits for TOKENS."
  (let ((photon-autograd--tape nil))
    (let* ((logits (nl-llm-agent--p5-forward model tokens))
           (tensor (pav-value logits)))
      (copy-sequence (photon-tensor-data tensor)))))

(defun arrt--tokens (length)
  "Return a bounded deterministic token list of LENGTH bytes."
  (unless (and (integerp length) (> length 0) (<= length 4096))
    (error "invalid test sequence length %S" length))
  (let ((index 0) result)
    (while (< index length)
      (push (mod (+ 19 (* index 37)) 256) result)
      (setq index (1+ index)))
    (nreverse result)))

(defun arrt--s0 (length dim seed sigma)
  "Return a detached deterministic nonzero S0 PAV."
  (photon-autograd-const
   (photon-tensor
    (list length dim)
    (nl-llm-recur-randn (* length dim) sigma seed))))

(defun arrt--all-logits (model tokens r s0)
  "Return detached full logits for MODEL and TOKENS."
  (let ((photon-autograd--tape nil))
    (let* ((forward (nl-llm-recur-forward model tokens r :k 0 :s0 s0))
           (tensor (pav-value (plist-get forward :logits))))
      (list (copy-sequence (photon-tensor-shape tensor))
            (copy-sequence (photon-tensor-data tensor))))))

(defun arrt--last-row (shape data)
  "Copy the final vocabulary row from SHAPE and DATA."
  (let* ((rows (car shape))
         (cols (cadr shape))
         (start (* (1- rows) cols)))
    (unless (and (> rows 0) (>= (length data) (+ start cols)))
      (error "invalid logits fixture"))
    (let ((row (make-vector cols 0.0)) (index 0))
      (while (< index cols)
        (aset row index (aref data (+ start index)))
        (setq index (1+ index)))
      row)))

(defun arrt--maxdiff (left right)
  "Return maximum absolute difference between equal-length vectors."
  (unless (= (length left) (length right))
    (error "logit vectors have different lengths"))
  (let ((maximum 0.0) (index 0))
    (while (< index (length left))
      (let ((difference (abs (- (aref left index) (aref right index)))))
        (when (or (/= difference difference) (> difference maximum))
          (setq maximum difference)))
      (setq index (1+ index)))
    maximum))

(defun arrt--parameter-snapshot (model &optional gradients)
  "Return detached MODEL parameter values, and optionally gradients."
  (mapcar
   (lambda (parameter)
     (list (copy-sequence (photon-tensor-shape (pav-value parameter)))
           (copy-sequence (photon-tensor-data (pav-value parameter)))
           (and gradients
                (copy-sequence (photon-tensor-data (pav-grad parameter))))))
   (nl-llm-recur-params model)))

(defun arrt--snapshot-equal-p (left right)
  "Return non-nil when two parameter snapshots are bit-identical."
  (and (= (length left) (length right))
       (cl-every
        (lambda (pair)
          (and (equal (nth 0 (car pair)) (nth 0 (cdr pair)))
               (equal (nth 1 (car pair)) (nth 1 (cdr pair)))
               (equal (nth 2 (car pair)) (nth 2 (cdr pair)))))
        (cl-mapcar #'cons left right))))

(defun arrt--provider-state (model r seed)
  "Build the private state shape consumed by provider LOGITS."
  (list :bundle (list :model model :r r :s0-seed seed)))

(defun arrt--provider-logits (model tokens r seed)
  "Run the provider's full-prefix CPU logits entry point."
  (nl-llm-agent-recur-provider--logits
   (arrt--provider-state model r seed) tokens))

(defun arrt--error-p (thunk)
  "Return the error string signaled by THUNK, or nil."
  (condition-case err
      (progn (funcall thunk) nil)
    (error (error-message-string err))))

(let* ((model (arrt--model))
       (params (nl-llm-recur-params model))
       ;; Nonzero sentinels make a cleared or replaced gradient observable.
       (_ (dolist (parameter params)
            (fillarray (photon-tensor-data (pav-grad parameter)) 0.375)))
       (value-before (arrt--parameter-snapshot model t))
       (runtime-targets nl-llm-inference-runtime--targets)
       (runtime-targets-present
        (and (memq 'photon-tensor-matmul runtime-targets)
             (memq 'photon-tensor-softmax-rows runtime-targets)
             (memq 'photon-tensor-scale runtime-targets)))
       (initial-bindings
        (mapcar (lambda (symbol) (cons symbol (symbol-function symbol)))
                (delete-dups
                 (append runtime-targets
                         nl-llm-inference-runtime--dependencies))))
       (initial-properties
        (mapcar (lambda (symbol)
                  (cons symbol (copy-tree (symbol-plist symbol))))
                (delete-dups
                 (append runtime-targets
                         nl-llm-inference-runtime--dependencies)))))
  (unwind-protect
      (progn
        (arrt--check "runtime target set includes recurrent hot kernels"
                     runtime-targets-present)
        (let ((source-results nil) (compiled-results nil))
          ;; Keep the same model and explicit nonzero S0 for every pair.  The
          ;; provider entry point also exercises its second CPU/GPU guard and
          ;; its runtime prepare boundary.
          (dolist (length '(1 9 33))
            (dolist (r '(1 2 4))
              (let* ((tokens (arrt--tokens length))
                     (seed (+ 7000 (* r 100) length))
                     (s0 (arrt--s0 length (plist-get model :dim) seed
                                  (plist-get model :sigma)))
                     (source
                     (let ((nl-llm-inference-runtime-mode 'source))
                        (nl-llm-inference-runtime-prepare 'source)
                        (when (and (= length 1) (= r 1))
                          (arrt--check "source mode retains source hot-kernel function"
                                       (not (byte-code-function-p
                                             (symbol-function
                                              'photon-tensor-matmul)))))
                        (arrt--all-logits model tokens r s0)))
                     (compiled
                      (let ((nl-llm-inference-runtime-mode 'byte-code))
                        (nl-llm-inference-runtime-prepare 'byte-code)
                        (when (and (= length 1) (= r 1))
                          (arrt--check "byte-code mode installs byte-compiled hot kernel"
                                       (byte-code-function-p
                                        (symbol-function
                                         'photon-tensor-matmul))))
                        (arrt--all-logits model tokens r s0))))
                (push source source-results)
                (push compiled compiled-results)
                (arrt--check
                 (format "source/byte-code parity length=%d R=%d" length r)
                 (= (arrt--maxdiff (cadr source) (cadr compiled)) 0.0)
                 (format "maxdiff=%.3e"
                         (arrt--maxdiff (cadr source) (cadr compiled)))))))
          ;; Compare the provider's last-row result with the independently
          ;; collected source logits, including a nonzero S0 seed.
          (let* ((length 9) (r 2) (tokens (arrt--tokens length)) (seed 8123)
                 (s0 (arrt--s0 length (plist-get model :dim) seed
                              (plist-get model :sigma)))
                 (direct (let ((nl-llm-inference-runtime-mode 'source))
                           (nl-llm-inference-runtime-prepare 'source)
                           (arrt--all-logits model tokens r s0)))
                 (provided
                  (let ((nl-llm-inference-runtime-mode 'source))
                    ;; The provider constructs this same S0 from seed.
                    (arrt--provider-logits model tokens r seed))))
            (arrt--check "provider returns exact deterministic final row"
                         (= (arrt--maxdiff provided
                                          (arrt--last-row (car direct)
                                                          (cadr direct)))
                            0.0)))
          ;; The complete runtime group must be restored before fallback checks.
          (nl-llm-inference-runtime-prepare 'source)
          (let ((original-byte-compile (and (fboundp 'byte-compile)
                                            (symbol-function 'byte-compile))))
            (cl-letf (((symbol-function 'byte-compile) nil))
              (arrt--check "auto falls back to source without compiler"
                           (eq (nl-llm-inference-runtime-prepare 'auto) 'source)))
            (when original-byte-compile
              (cl-letf (((symbol-function 'byte-compile)
                         (lambda (&rest _args)
                           (error "injected runtime compiler failure"))))
                (arrt--check "auto falls back transactionally on compiler error"
                             (eq (nl-llm-inference-runtime-prepare 'auto) 'source))))))

        ;; A redefined hot target must be called by refreshed byte-code and
        ;; remain the external definition when source mode restores ownership.
        (nl-llm-inference-runtime-prepare 'source)
        (let* ((target 'photon-tensor-matmul)
               (original (symbol-function target))
               (calls 0)
               (replacement (lambda (a b)
                              (setq calls (1+ calls))
                              (funcall original a b))))
          (unwind-protect
              (progn
                (fset target replacement)
                (arrt--check "auto rejects captured target closure transactionally"
                             (eq (nl-llm-inference-runtime-prepare 'auto)
                                 'source))
                (arrt--all-logits model (arrt--tokens 9) 2
                                  (arrt--s0 9 (plist-get model :dim) 9101
                                            (plist-get model :sigma)))
                (arrt--check "auto fallback retains callable external matmul closure"
                             (> calls 0) (format "calls=%d" calls))
                (let ((calls-before calls))
                  (arrt--check "explicit byte-code rejects captured target closure"
                               (and (arrt--error-p
                                     (lambda ()
                                       (nl-llm-inference-runtime-prepare
                                        'byte-code)))
                                    (= calls calls-before))))
                (nl-llm-inference-runtime-prepare 'source)
                (arrt--check "source restore retains external matmul definition"
                             (eq (symbol-function target) replacement)))
            (nl-llm-inference-runtime-prepare 'source)
            (fset target original)))

        ;; CPU/GPU aliases are simulated with function cells; no GPU function is
        ;; executed.  The first guard must reject before prepare or tensor work.
        (let* ((target 'photon-tensor-matmul)
               (gpu-target 'photon-tensor-matmul-gpu)
               (original (symbol-function target))
               (gpu-was-bound (fboundp gpu-target))
               (gpu-original (and gpu-was-bound
                                  (symbol-function gpu-target)))
               (gpu-calls 0)
               (gpu (lambda (&rest _args)
                      (setq gpu-calls (1+ gpu-calls))
                      (error "GPU stub must not run"))))
          (unwind-protect
              (progn
                (fset gpu-target gpu)
                (fset target gpu)
                (arrt--check "active GPU alias rejects before tensor calls"
                             (let ((message (arrt--error-p
                                             (lambda ()
                                               (arrt--provider-logits
                                                model (arrt--tokens 1) 1 991)))))
                               (and message
                                    (string-match-p "active GPU" message)
                                    (= gpu-calls 0)))))
            (fset target original)
            (if gpu-was-bound
                (fset gpu-target gpu-original)
              (fmakunbound gpu-target))))

        ;; The second guard catches a compiler hook which swaps a backend after
        ;; the initial check but before the recurrent forward call.
        (let* ((target 'photon-tensor-matmul)
               (gpu-target 'photon-tensor-matmul-gpu)
               (original (symbol-function target))
               (gpu-was-bound (fboundp gpu-target))
               (gpu-original (and gpu-was-bound
                                  (symbol-function gpu-target)))
               (gpu-calls 0)
               (gpu (lambda (&rest _args)
                      (setq gpu-calls (1+ gpu-calls))
                      (error "GPU stub must not run"))))
          (unwind-protect
              (progn
                (fset gpu-target gpu)
                (fset target original)
                (cl-letf (((symbol-function 'nl-llm-inference-runtime-prepare)
                           (lambda (&optional _mode)
                             (fset target gpu)
                             'byte-code)))
                  (arrt--check "compiler-hook GPU swap rejects before tensor calls"
                               (let ((message (arrt--error-p
                                               (lambda ()
                                                 (arrt--provider-logits
                                                  model (arrt--tokens 1) 1 992)))))
                                 (and message
                                      (string-match-p "active GPU" message)
                                      (= gpu-calls 0))))))
            (fset target original)
            (if gpu-was-bound
                (fset gpu-target gpu-original)
              (fmakunbound gpu-target))))

        ;; Exercise the real compiler transaction: a byte-compile hook changes
        ;; a target after compilation, so auto preparation must fall back to
        ;; source and the provider's second backend guard must reject it.
        (let* ((target 'photon-tensor-matmul)
               (gpu-target 'photon-tensor-matmul-gpu)
               (original (symbol-function target))
               (gpu-was-bound (fboundp gpu-target))
               (gpu-original (and gpu-was-bound
                                  (symbol-function gpu-target)))
               (compiler (and (fboundp 'byte-compile)
                              (symbol-function 'byte-compile)))
               (compile-calls 0)
               (gpu-calls 0)
               (gpu (lambda (&rest _args)
                      (setq gpu-calls (1+ gpu-calls))
                      (error "GPU stub must not run"))))
          (unwind-protect
              (if (not compiler)
                  (arrt--check "real compiler-hook requires Emacs byte compiler"
                               nil "compiler unavailable")
                (progn
                  (fset gpu-target gpu)
                  (fset target original)
                  (cl-letf (((symbol-function 'byte-compile)
                             (lambda (definition)
                               (setq compile-calls (1+ compile-calls))
                               (let ((compiled (funcall compiler definition)))
                                 (when (= compile-calls
                                          (length
                                           nl-llm-inference-runtime--targets))
                                   (fset target gpu))
                                 compiled))))
                    (let ((nl-llm-inference-runtime-mode 'auto))
                      (arrt--check "actual compiler-hook GPU swap is rejected"
                                   (let ((message (arrt--error-p
                                                   (lambda ()
                                                     (arrt--provider-logits
                                                      model (arrt--tokens 1)
                                                      1 993)))))
                                     (and message
                                          (string-match-p "active GPU" message)
                                          (> compile-calls 0)
                                          (= gpu-calls 0))))))))
            (nl-llm-inference-runtime-prepare 'source)
            (fset target original)
            (if gpu-was-bound
                (fset gpu-target gpu-original)
              (fmakunbound gpu-target))))

        ;; The provider uses an explicit S0 and an isolated no-tape scope.  It
        ;; must not mutate parameters, their gradients, the caller's tape, or
        ;; Emacs' global random stream.
        (let* ((tokens (arrt--tokens 9)) (seed 9007)
               (tape-marker (list 'caller-tape))
               expected-random actual-random)
          (random "agent-recur-runtime-random-sentinel")
          (setq expected-random (list (random) (random)))
          (random "agent-recur-runtime-random-sentinel")
          (let ((photon-autograd--tape tape-marker))
            (arrt--provider-logits model tokens 2 seed)
            (arrt--check "provider leaves caller autograd tape untouched"
                         (eq photon-autograd--tape tape-marker)))
          (setq actual-random (list (random) (random)))
          (arrt--check "provider leaves global random stream untouched"
                       (equal expected-random actual-random))
          (arrt--check "provider leaves parameter values and gradients untouched"
                       (arrt--snapshot-equal-p value-before
                                                (arrt--parameter-snapshot model t))))

        ;; Interleaving the shared runtime with a recurrent call must not leave
        ;; dense targets half-installed; the existing dense runtime owns the
        ;; same transaction and source restore boundary.
        (let* ((before (mapcar (lambda (entry) (cons (car entry) (cdr entry)))
                               initial-bindings))
               (dense (arrt--dense-model))
               (dense-tokens '(3 19 65 127))
               dense-source dense-byte)
          (nl-llm-inference-runtime-prepare 'source)
          (setq dense-source (arrt--dense-logits dense dense-tokens))
          (nl-llm-inference-runtime-prepare 'byte-code)
          (setq dense-byte (arrt--dense-logits dense dense-tokens))
          (arrt--check "dense path remains numerically unchanged across shared runtime"
                       (= (arrt--maxdiff dense-source dense-byte)
                          0.0))
          (arrt--all-logits model (arrt--tokens 1) 1
                            (arrt--s0 1 (plist-get model :dim) 9301
                                      (plist-get model :sigma)))
          (nl-llm-inference-runtime-prepare 'source)
          (arrt--check "dense/recurrent shared runtime source restore is complete"
                       (cl-every
                        (lambda (entry)
                          (eq (symbol-function (car entry)) (cdr entry)))
                        before)))
        (arrt--check "model has a nonempty parameter set" (> (length params) 0)))
    (nl-llm-inference-runtime-prepare 'source)
    (dolist (entry initial-bindings)
      (fset (car entry) (cdr entry)))
    (dolist (entry initial-properties)
      (setplist (car entry) (cdr entry)))))

(princ (format "NL-LLM-AGENT-RECUR-RUNTIME %s (%d failures)\n"
               (if (= arrt--fail 0) "ALL-PASS" "HAS-FAILURES") arrt--fail))
(kill-emacs (if (= arrt--fail 0) 0 1))

;;; agent-recur-runtime-test.el ends here
