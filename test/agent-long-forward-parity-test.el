;;; agent-long-forward-parity-test.el --- long GPU/native forward parity -*- lexical-binding: t; -*-

;; This file intentionally does not load the original autorun parity test.
;; Load with `nl-llm-agent-long-forward-no-run' non-nil to inspect the helpers
;; and fixture without starting a GPU test.

(defvar nl-llm-agent-long-forward-no-run nil
  "When non-nil, load this file without running its ERT entry point.")

(defconst alfp--here
  (file-name-directory (or load-file-name buffer-file-name default-directory)))
(setq load-prefer-newer t)
(add-to-list 'load-path (expand-file-name "../lisp" alfp--here))
(add-to-list 'load-path (expand-file-name "../../nelisp-photon/lisp" alfp--here))
(require 'cl-lib)
(require 'ert)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-agent-artifact)
(require 'nl-llm-agent-improve)
(require 'nl-llm-agent-initialization)
(require 'nl-llm-agent-tokenizer)
(require 'nl-llm-decode)
(require 'nl-llm-gpu)
(require 'nl-llm-gpu-ag)
(require 'nl-llm-inference-runtime)

(defconst alfp--real-rows 1476)
(defconst alfp--physical-rows 2048)
(defconst alfp--vocab 256)
(defconst alfp--dim 32)
(defconst alfp--ff 64)
(defconst alfp--blocks 1)
(defconst alfp--heads 1)
(defconst alfp--seed 439041101)
(defconst alfp--gpu-absolute-tolerance 5.0e-4)
(defconst alfp--gpu-relative-tolerance 5.0e-4)
(defconst alfp--padding-tolerance 1.0e-6)
(defconst alfp--native-tolerance 1.0e-10)

(defconst alfp--block-keys
  '(:ln1g :wq :bq :wk :bk :wv :bv :wo :bo
    :ln2g :wg :bg :wu :bu :wd :bd))

(defun alfp--finite-vector-p (values)
  "Return non-nil when every element of VALUES is finite numeric data."
  (let ((finite t))
    (dotimes (index (length values))
      (let ((value (aref values index)))
        (unless (and (numberp value)
                     (= value value)
                     (<= (abs value) 1.7976931348623157e+308))
          (setq finite nil))))
    finite))

(defun alfp--assert-cpu-dispatch ()
  "Assert that every GPU-overridden tensor op points to its saved CPU cell."
  (unless (and (boundp 'photon-tensor-gpu--saved)
               photon-tensor-gpu--saved)
    (error "long forward parity has no saved CPU tensor dispatch"))
  (dolist (pair photon-tensor-gpu--saved)
    (unless (eq (symbol-function (car pair)) (cdr pair))
      (error "long forward parity tensor op is not on CPU dispatch: %S"
             (car pair))))
  t)

(defun alfp--fixture-text ()
  "Return the deterministic valid UTF-8 fixture of exactly 1476 bytes."
    (let* ((pattern "train/status.txt | docs/agent | A/B: 日🙂 | ASCII\n")
         (pattern-tokens
          (nl-llm-agent-tokenizer-encode pattern "utf8-byte-v1"))
         (pattern-bytes (length pattern-tokens))
         (text "")
         (bytes 0))
    (while (<= (+ bytes pattern-bytes) alfp--real-rows)
      (setq text (concat text pattern)
            bytes (+ bytes pattern-bytes)))
    (concat text (make-string (- alfp--real-rows bytes) ?A))))

(defun alfp--fixture ()
  "Return the fixture text and byte-token vector after cheap validation."
  (let* ((text (alfp--fixture-text))
         (tokens
          (apply #'vector
                 (nl-llm-agent-tokenizer-encode text "utf8-byte-v1"))))
    (unless (= (string-bytes text) alfp--real-rows)
      (error "long forward fixture has %d bytes, expected %d"
             (string-bytes text) alfp--real-rows))
    (unless (= (length tokens) alfp--real-rows)
      (error "long forward fixture has %d tokens, expected %d"
             (length tokens) alfp--real-rows))
    (dolist (needle '("train/" "docs/" "A/B" "日" "🙂"))
      (unless (string-match-p (regexp-quote needle) text)
        (error "long forward fixture is missing %S" needle)))
    (dotimes (index (length tokens))
      (unless (and (integerp (aref tokens index))
                   (<= 0 (aref tokens index))
                   (< (aref tokens index) alfp--vocab))
        (error "long forward fixture token %d is invalid: %S"
               index (aref tokens index))))
    (list :text text :tokens tokens)))

(defun alfp--padded-tokens (tokens pad)
  "Return TOKENS followed by PAD to the physical 2048-row GPU window."
  (unless (= (length tokens) alfp--real-rows)
    (error "cannot pad %d real rows" (length tokens)))
  (let ((result (make-vector alfp--physical-rows pad)))
    (dotimes (index alfp--real-rows)
      (aset result index (aref tokens index)))
    result))

(defun alfp--parameter-values (model)
  "Return detached parameter arrays in canonical P5 order."
  (mapcar
   (lambda (parameter)
     (copy-sequence (photon-tensor-data (pav-value parameter))))
   (nl-llm-agent--p5-params model)))

(defun alfp--model-geometry-ok (model)
  "Signal unless MODEL has the requested exact geometry."
  (dolist (pair `((:dim . ,alfp--dim) (:ff . ,alfp--ff)
                  (:vocab . ,alfp--vocab) (:nblocks . ,alfp--blocks)
                  (:heads . ,alfp--heads)))
    (unless (= (plist-get model (car pair)) (cdr pair))
      (error "model geometry %S is %S, expected %S"
             (car pair) (plist-get model (car pair)) (cdr pair))))
  (unless (equal (plist-get model :tokenizer) "utf8-byte-v1")
    (error "model tokenizer is %S" (plist-get model :tokenizer)))
  t)

(defun alfp--perturb (model)
  "Apply the deterministic nonzero short-forward perturbations to MODEL."
  (cl-labels
      ((add (pav index delta)
         (let ((data (photon-tensor-data (pav-value pav))))
           (aset data index (+ (aref data index) delta)))))
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

(defun alfp--native-prefill (model tokens mode label)
  "Run one exported native prefill in MODE with a 2048-row cache."
  (let ((prepared (nl-llm-inference-runtime-prepare mode)))
    (unless (eq prepared mode)
      (error "native %s requested %S but prepared %S" label mode prepared))
    (when (eq mode 'source)
      (alfp--assert-cpu-dispatch))
    (when (eq mode 'byte-code)
      (alfp--runtime-byte-code-ok))
    (let* ((checkpoint (nl-llm-agent-artifact-export-pav model))
           (exported
            (nl-llm-agent-artifact--checkpoint-model
             (append (list :format nl-llm-ckpt-format) checkpoint)
             (format "long-forward-%s" label)))
           (dim (plist-get exported :dim))
           (heads (plist-get exported :heads))
           (kvh (plist-get exported :kvh))
           (caches
            (mapcar
             (lambda (_block)
               (nl-llm-dcache-new alfp--physical-rows dim heads kvh))
             (plist-get exported :blocks)))
           (rows nil))
      (dotimes (index (length tokens))
        (push
         (nl-llm-decode-step
          (aref tokens index) (plist-get exported :blocks) caches
          (plist-get exported :wte) (plist-get exported :lnfg)
          (plist-get exported :bh) dim nil (plist-get exported :wh))
         rows)
        (when (zerop (mod (1+ index) 128))
          (princ (format "native %s %s position %d/%d\n"
                         label mode (1+ index) (length tokens)))))
      (apply #'vconcat (nreverse rows)))))

(defun alfp--runtime-byte-code-ok ()
  "Assert byte-code targets and the allowed GPU dispatch ownership split."
  (dolist (symbol nl-llm-inference-runtime--targets)
    (unless (byte-code-function-p (symbol-function symbol))
      (error "runtime target %s is not byte-code" symbol)))
  (dolist (pair photon-tensor-gpu--saved)
    (let* ((symbol (car pair))
           (cpu (cdr pair))
           (installed (symbol-function symbol))
           (source (assq symbol nl-llm-inference-runtime--sources))
           (owned (assq symbol nl-llm-inference-runtime--owned)))
      (unless (or (eq installed cpu)
                  (and (memq symbol nl-llm-inference-runtime--targets)
                       source owned
                       (eq (cdr source) cpu)
                       (eq installed (cdr owned))))
        (error "runtime byte-code dispatch ownership invalid for %S" symbol))))
  t)

(defun alfp--gpu-logits (model tokens pad label)
  "Return all physical GPU logit rows for MODEL with padding token PAD."
  (let ((builder nil)
        (padded (alfp--padded-tokens tokens pad)))
    (princ (format "GPU begin %s pad=%d rows=%d\n"
                   label pad alfp--physical-rows))
    (unwind-protect
        (progn
          (setq builder (nlga-new))
          (cl-labels
              ((parameter (pav)
                 (nlga-param builder (pav-value pav)))
               (block (source)
                 (let ((result nil))
                   (dolist (key alfp--block-keys)
                     (setq result
                           (append result
                                   (list key (parameter (plist-get source key))))))
                   result)))
            (let* ((token-rt
                    (nlga-const
                     builder
                     (photon-tensor
                      (list alfp--physical-rows 1)
                      (apply #'vector (mapcar #'float (append padded nil))))))
                   (wte (parameter (plist-get model :wte)))
                   (blocks (mapcar #'block (plist-get model :blocks)))
                   (lnfg (parameter (plist-get model :lnfg)))
                   (wh (parameter (plist-get model :wh)))
                   (bh (parameter (plist-get model :bh)))
                   (tables
                    (nl-llm-gpu-rope-tables
                     alfp--physical-rows (/ alfp--dim alfp--heads)))
                   (cosr (nlga-const builder (car tables)))
                   (sinr (nlga-const builder (cdr tables)))
                   (spos (nlga-scalar builder 1.0))
                   (sneg (nlga-scalar builder -1.0))
                   (one (nlga-scalar builder 1.0))
                   (logits
                    (nlga-model-idx
                     builder token-rt wte blocks lnfg wh bh
                     alfp--heads alfp--heads cosr sinr spos sneg nil nil))
                   (output (nlga-keep builder logits one)))
              ;; Forward only: no loss seed, backward pass, optimizer, or readback.
              (nlga-compile builder)
              (let ((result (copy-sequence (nth output (nlga-step builder)))))
                (unless (= (length result) (* alfp--physical-rows alfp--vocab))
                  (error "GPU %s output has %d values, expected %d"
                         label (length result)
                         (* alfp--physical-rows alfp--vocab)))
                (unless (alfp--finite-vector-p result)
                  (error "GPU %s output contains non-finite data" label))
                result))))
      (when builder
        (nlga-free builder))
      (princ (format "GPU end %s pad=%d\n" label pad)))))

(defun alfp--compare-rows (left right rows vocab absolute relative-tolerance label)
  "Compare ROWS rows of LEFT and RIGHT and return a diagnostic plist."
  (let ((max-absolute 0.0)
        (max-relative 0.0)
        (violations 0)
        (count (* rows vocab)))
    (unless (and (= (length left) (* rows vocab))
                 (= (length right) (* rows vocab)))
      (error "%s vector lengths are %d and %d, expected %d"
             label (length left) (length right) count))
    (dotimes (index count)
      (let ((a (aref left index))
            (b (aref right index)))
        (unless (and (numberp a) (numberp b)
                     (= a a) (= b b)
                     (<= (abs a) 1.7976931348623157e+308)
                     (<= (abs b) 1.7976931348623157e+308))
          (error "%s contains non-finite or nonnumeric value at %d" label index))
        (let* ((difference (abs (- a b)))
               (relative-error (/ difference (max 1.0e-30 (abs b)))))
          (setq max-absolute (max max-absolute difference)
                max-relative (max max-relative relative-error))
          (when (> difference (+ absolute (* relative-tolerance (abs b))))
            (setq violations (1+ violations))))))
          (list :max-absolute max-absolute :max-relative max-relative
          :violations violations)))

(defun alfp--real-gpu-rows (values label)
  "Validate a physical GPU result and detach only its real-row prefix."
  (let ((physical (* alfp--physical-rows alfp--vocab))
        (real (* alfp--real-rows alfp--vocab)))
    (unless (= (length values) physical)
      (error "%s GPU result has %d values, expected physical %d"
             label (length values) physical))
    (unless (alfp--finite-vector-p values)
      (error "%s GPU result contains non-finite data" label))
    (let ((prefix (make-vector real 0.0)))
      (dotimes (index real)
        (aset prefix index (aref values index)))
      prefix)))

(defun alfp--cheap-self-checks ()
  "Exercise comparison failure and non-finite guards without a GPU."
  (let ((wrong
         (alfp--compare-rows [2.0] [1.0] 1 1 5.0e-4 5.0e-4
                             "cheap intentionally-wrong")))
    (unless (> (plist-get wrong :violations) 0)
      (error "cheap intentionally-wrong comparison did not fail")))
  (let ((rejected nil))
    (condition-case _err
        (alfp--compare-rows (vector 0.0 (log -1.0)) [0.0 0.0] 1 2
                            1.0e-6 0.0 "cheap non-finite")
      (error (setq rejected t)))
    (unless rejected
      (error "cheap non-finite comparison was accepted")))
  (let ((rejected nil))
    (condition-case _err
        (alfp--compare-rows (vector 0.0 (exp 10000.0)) [0.0 0.0] 1 2
                            1.0e-6 0.0 "cheap infinite")
      (error (setq rejected t)))
    (unless rejected
      (error "cheap infinite comparison was accepted")))
  t)

(defun alfp--report-comparison (label result)
  "Print RESULT from `alfp--compare-rows' with LABEL."
  (princ (format "%s maxabs=%.3g maxrel=%.3g violations=%d\n"
                 label (plist-get result :max-absolute)
                 (plist-get result :max-relative)
                 (plist-get result :violations))))

(ert-deftest nl-llm-agent-long-forward-parity ()
  "Compare long indexed GPU prefill with source and byte-code native paths."
  (let ((gpu-enabled nil))
    (alfp--cheap-self-checks)
    (unwind-protect
        (progn
          (setq gpu-enabled (nl-llm-gpu-enable))
          (unless gpu-enabled
            (ert-skip "SKIP: no Vulkan device"))
          ;; Direct NLGA remains GPU-backed, while every host tensor operation
          ;; used by export and native decode is pinned to its saved CPU cell.
          (photon-tensor-use-cpu-backend)
          (alfp--assert-cpu-dispatch)
          (let* ((fixture (alfp--fixture))
                 (tokens (plist-get fixture :tokens))
                 ;; Constructors are deliberately inside this ERT body: no
                 ;; model or task data is created merely by loading the file.
                 (initial
                  (nl-llm-agent-initialization-create
                   :initializer 'xorshift32 :seed alfp--seed
                   :dim alfp--dim :ff alfp--ff :vocab alfp--vocab
                   :nblocks alfp--blocks :heads alfp--heads
                   :tokenizer "utf8-byte-v1"))
                 (perturbed
                  (alfp--perturb
                   (nl-llm-agent-initialization-create
                    :initializer 'xorshift32 :seed alfp--seed
                    :dim alfp--dim :ff alfp--ff :vocab alfp--vocab
                   :nblocks alfp--blocks :heads alfp--heads
                   :tokenizer "utf8-byte-v1"))))
            (dolist (model (list (cons "initial" initial)
                                 (cons "perturbed" perturbed)))
              (let* ((variant (car model))
                     (value (cdr model))
                     (before (alfp--parameter-values value)))
                (alfp--model-geometry-ok value)
                (let ((source
                       (alfp--native-prefill value tokens 'source variant)))
                  (alfp--assert-cpu-dispatch)
                  (let ((byte-code
                         (alfp--native-prefill
                          value tokens 'byte-code variant)))
                    (alfp--runtime-byte-code-ok)
                    (should (equal (nl-llm-inference-runtime-prepare 'source)
                                   'source))
                    (alfp--assert-cpu-dispatch)
                    (let ((native-check
                           (alfp--compare-rows
                            source byte-code alfp--real-rows alfp--vocab
                            alfp--native-tolerance 0.0
                            (format "%s native" variant))))
                      (alfp--report-comparison
                       (format "%s native source/byte-code" variant)
                       native-check)
                      (should (= (plist-get native-check :violations) 0)))
                    (let* ((gpu-zero
                            (alfp--real-gpu-rows
                             (alfp--gpu-logits value tokens 0 variant)
                             (format "%s pad0" variant)))
                           (gpu-255
                            (alfp--real-gpu-rows
                             (alfp--gpu-logits value tokens 255 variant)
                             (format "%s pad255" variant))))
                      (dolist (gpu-case (list (cons 0 gpu-zero)
                                              (cons 255 gpu-255)))
                        (let* ((pad (car gpu-case))
                               (gpu (cdr gpu-case))
                             (source-check
                              (alfp--compare-rows
                               gpu source alfp--real-rows alfp--vocab
                               alfp--gpu-absolute-tolerance
                               alfp--gpu-relative-tolerance
                               (format "%s GPU pad=%d/source" variant pad)))
                             (byte-check
                              (alfp--compare-rows
                               gpu byte-code alfp--real-rows alfp--vocab
                               alfp--gpu-absolute-tolerance
                               alfp--gpu-relative-tolerance
                               (format "%s GPU pad=%d/byte-code" variant pad))))
                        (alfp--report-comparison
                         (format "%s GPU pad=%d/source" variant pad)
                         source-check)
                        (alfp--report-comparison
                         (format "%s GPU pad=%d/byte-code" variant pad)
                         byte-check)
                        (should (= (plist-get source-check :violations) 0))
                        (should (= (plist-get byte-check :violations) 0))))
                      (let ((padding-check
                             (alfp--compare-rows
                              gpu-zero gpu-255 alfp--real-rows alfp--vocab
                              alfp--padding-tolerance 0.0
                              (format "%s GPU pad0/pad255" variant))))
                      (alfp--report-comparison
                       (format "%s GPU real rows pad0/pad255" variant)
                       padding-check)
                        (should (= (plist-get padding-check :violations) 0)))))
                (should (equal before (alfp--parameter-values value))))))))
      ;; Restore runtime-owned byte-code before returning to GPU teardown.
      (ignore-errors (nl-llm-inference-runtime-prepare 'source))
      (when gpu-enabled
        (nl-llm-gpu-disable)))))

(unless nl-llm-agent-long-forward-no-run
  (ert-run-tests-batch-and-exit 'nl-llm-agent-long-forward-parity))

;;; agent-long-forward-parity-test.el ends here
