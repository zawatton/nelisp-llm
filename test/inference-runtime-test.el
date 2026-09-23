;;; inference-runtime-test.el --- in-memory inference compiler tests  -*- lexical-binding: t; -*-

;;   emacs -Q --batch \
;;     -l test/inference-runtime-test.el

(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-decode)
(require 'nl-llm-inference-runtime)
(load (expand-file-name "test/interpreted-targets.el") nil t t)

(defvar irt--fail 0)

(defun irt--check (name ok &optional detail)
  (princ (format "%-55s %s  %s\n" name
                 (if ok "PASS" (setq irt--fail (1+ irt--fail)) "FAIL")
                 (or detail ""))))

(defun irt--tensor (shape seed scale)
  (let ((size 1))
    (dolist (dimension shape) (setq size (* size dimension)))
    (photon-tensor
     shape
     (let ((data (make-vector size 0.0)) (i 0))
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
       data))))

(defun irt--constant (n value)
  (photon-tensor (list n) (make-vector n value)))

(defun irt--model ()
  (let* ((dim 4) (ff 4) (vocab 8) (scale 0.5)
         (block
          (list :ln1g (irt--constant dim 1.0)
                :wq (irt--tensor (list dim dim) 21 scale)
                :bq (irt--constant dim 0.0)
                :wk (irt--tensor (list dim dim) 22 scale)
                :bk (irt--constant dim 0.0)
                :wv (irt--tensor (list dim dim) 23 scale)
                :bv (irt--constant dim 0.0)
                :wo (irt--tensor (list dim dim) 24 scale)
                :bo (irt--constant dim 0.0)
                :ln2g (irt--constant dim 1.0)
                :wg (irt--tensor (list ff dim) 25 scale)
                :bg (irt--constant ff 0.0)
                :wu (irt--tensor (list ff dim) 26 scale)
                :bu (irt--constant ff 0.0)
                :wd (irt--tensor (list dim ff) 27 scale)
                :bd (irt--constant dim 0.0))))
    (list :blocks (list block)
          :wte (irt--tensor (list vocab dim) 1 scale)
          :wh (irt--tensor (list vocab dim) 9 scale)
          :lnfg (irt--constant dim 1.0)
          :bh (irt--constant vocab 0.0)
          :dim dim :heads 1 :kvh 1)))

(defun irt--decode (model tokens)
  (let* ((dim (plist-get model :dim))
         (blocks (plist-get model :blocks))
         (caches (mapcar (lambda (_block)
                           (nl-llm-dcache-new (length tokens) dim 1 1))
                         blocks))
         (result nil))
    (dolist (token tokens)
      (push (nl-llm-decode-step
             token blocks caches (plist-get model :wte)
             (plist-get model :lnfg) (plist-get model :bh) dim nil
             (plist-get model :wh))
            result))
    (nreverse result)))

(defun irt--max-difference (left right)
  (let ((maximum 0.0))
    (while left
      (let ((a (car left)) (b (car right)) (i 0))
        (while (< i (length a))
          (setq maximum (max maximum (abs (- (aref a i) (aref b i)))))
          (setq i (1+ i))))
      (setq left (cdr left) right (cdr right)))
    maximum))

(defun irt--bindings (symbols)
  (mapcar (lambda (symbol) (cons symbol (symbol-function symbol))) symbols))

(defun irt--bindings-match-p (bindings)
  (cl-every (lambda (entry)
              (and (fboundp (car entry))
                   (eq (symbol-function (car entry)) (cdr entry))))
            bindings))

;; This module compiles the inference path in memory with inlining inhibited,
;; so that a dependency redefined between turns is honoured.  That guarantee is
;; only expressible when its targets reach it interpreted: `byte-compile' hands
;; an existing byte-code object straight back, so a target loaded from a .elc
;; keeps whatever defsubst bodies were inlined into it at file-compile time.
;;
;; Both worlds are real -- a checkout with lisp/*.elc built and one without --
;; so both are checked here.  The refusal is checked first, on whatever state
;; the tree is actually in; then the target files are reloaded from source, so
;; the rest of the suite exercises in-memory compilation as it always has,
;; whether or not a .elc exists.
(defvar irt--precompiled (interpreted-targets-precompiled))

(if (null irt--precompiled)
    (princ (format "%-55s %s  %s\n" "pre-compiled targets" "n/a"
                   "tree is not byte-compiled; that path not exercised"))
  ;; A caller asking for byte-code wants compiled numeric kernels and, on a
  ;; built tree, already has them -- so preparation reports `byte-code' and
  ;; owns nothing, rather than either refusing or claiming an object it did
  ;; not make.  What it cannot offer is the refresh guarantee, which is why
  ;; the suite below reloads from source before testing that.
  (irt--check "pre-compiled targets: byte-code mode accepts them"
              (eq (nl-llm-inference-runtime-prepare 'byte-code) 'byte-code)
              (format "%d of %d targets arrived byte-compiled"
                      (length irt--precompiled)
                      (length nl-llm-inference-runtime--targets)))
  (irt--check "pre-compiled targets: auto agrees"
              (eq (nl-llm-inference-runtime-prepare 'auto) 'byte-code))
  (irt--check "pre-compiled targets: nothing is owned, nothing is clobbered"
              (and (null nl-llm-inference-runtime--owned)
                   (progn (nl-llm-inference-runtime-prepare 'source)
                          (byte-code-function-p
                           (symbol-function (car irt--precompiled))))))
  ;; Control: a group that is part compiled and part source cannot be prepared
  ;; transactionally, and saying so beats silently preparing half of it.
  (let* ((target (car nl-llm-inference-runtime--targets))
         (original (symbol-function target)))
    (unwind-protect
        (progn
          (fset target (eval '(lambda (&rest args) (car args)) t))
          (irt--check "a part-compiled, part-source group is refused"
                      (condition-case err
                          (progn (nl-llm-inference-runtime-prepare 'byte-code)
                                 nil)
                        (error (and (string-match-p
                                     "part compiled, part source"
                                     (error-message-string err))
                                    t)))))
      (fset target original))))

(irt--check "target files reloaded from source"
            (null (interpreted-targets-reload))
            "every target is interpreted again")

(let* ((all-symbols
        (delete-dups
         (append nl-llm-inference-runtime--targets
                 nl-llm-inference-runtime--dependencies)))
       (initial-bindings (irt--bindings all-symbols))
       (initial-properties
        (mapcar (lambda (symbol)
                  (cons symbol (copy-tree (symbol-plist symbol))))
                all-symbols)))
  (unwind-protect
      (let* ((model (irt--model))
             (tokens '(0 3 1 4 2 5 3 6))
             source-logits compiled-logits)
        (nl-llm-inference-runtime-prepare 'source)
        (setq source-logits (irt--decode model tokens))
        (irt--check "explicit source selects source"
                    (eq (nl-llm-inference-runtime-prepare 'source) 'source))

        (irt--check "cold default auto selects byte-code"
                    (eq (nl-llm-inference-runtime-prepare)
                        'byte-code))
        (irt--check "explicit byte-code selects byte-code"
                    (eq (nl-llm-inference-runtime-prepare 'byte-code)
                        'byte-code))
        (setq compiled-logits (irt--decode model tokens))
        (let ((difference
               (irt--max-difference source-logits compiled-logits)))
          (irt--check "source/compiled numeric parity at all positions"
                      (= difference 0.0) (format "maxdiff=%.3e" difference)))

        (let ((installed (symbol-function 'nl-llm-decode-block)))
          (nl-llm-inference-runtime-prepare 'byte-code)
          (irt--check "unchanged prepare reuses owned byte-code"
                      (eq installed (symbol-function 'nl-llm-decode-block))))

        (nl-llm-inference-runtime-prepare 'source)
        (irt--check "source mode restores original target bindings"
                    (irt--bindings-match-p
                     (cl-remove-if-not
                      (lambda (entry)
                        (memq (car entry)
                              nl-llm-inference-runtime--targets))
                      initial-bindings)))

        ;; Replacing an owned target with a capture-free source function is an
        ;; external write.  Refresh must compile the complete group and source
        ;; mode must restore that exact external definition.  Captured lexical
        ;; closures are deliberately tested by the agent recurrent runtime
        ;; test, where they must fall back to source instead.
        (nl-llm-inference-runtime-prepare 'byte-code)
        (let* ((original (cdr (assq 'nl-llm-silu initial-bindings)))
               (old-peer (symbol-function 'nl-llm-decode-block))
               (replacement (eval '(lambda (&rest args) (car args)) t)))
          (fset 'nl-llm-silu replacement)
          (nl-llm-inference-runtime-prepare 'byte-code)
          (irt--check "hot target redefine recompiles the complete group"
                      (not (eq old-peer
                               (symbol-function 'nl-llm-decode-block))))
          (nl-llm-inference-runtime-prepare 'source)
          (irt--check "source restore preserves external target redefine"
                      (eq replacement (symbol-function 'nl-llm-silu)))
          (fset 'nl-llm-silu original))

        ;; Defsubst dependencies can otherwise be silently embedded.  With
        ;; optimization inhibited, a refreshed compiled decoder must call the
        ;; external accessor and expose its observable side effect.
        (nl-llm-inference-runtime-prepare 'byte-code)
        (let* ((dependency 'photon-tensor-data)
               (original (cdr (assq dependency initial-bindings)))
               (properties (copy-tree (symbol-plist dependency)))
               (old-target (symbol-function 'nl-llm-decode-block))
               (calls 0)
               (replacement
                (lambda (tensor)
                  (setq calls (1+ calls))
                  (funcall original tensor))))
          (unwind-protect
              (progn
                (fset dependency replacement)
                (nl-llm-inference-runtime-prepare 'byte-code)
                (irt--check "dependency redefine refreshes complete group"
                            (not (eq old-target
                                     (symbol-function
                                      'nl-llm-decode-block))))
                (irt--decode model '(1 2))
                (irt--check "compiled caller honors redefined dependency"
                            (> calls 0) (format "calls=%d" calls))
                (nl-llm-inference-runtime-prepare 'source)
                (irt--check "dependency remains externally owned"
                            (eq replacement (symbol-function dependency))))
            (fset dependency original)
            (setplist dependency properties)))

        ;; Compiler metadata alone is also part of the reuse fingerprint.
        (nl-llm-inference-runtime-prepare 'byte-code)
        (let* ((dependency 'photon-tensor-data)
               (properties (copy-tree (symbol-plist dependency)))
               (old-target (symbol-function 'nl-llm-decode-block)))
          (unwind-protect
              (progn
                (put dependency 'byte-optimizer nil)
                (nl-llm-inference-runtime-prepare 'byte-code)
                (irt--check "dependency property edit refreshes complete group"
                            (not (eq old-target
                                     (symbol-function
                                      'nl-llm-decode-block)))))
            (nl-llm-inference-runtime-prepare 'source)
            (setplist dependency properties)))

        (nl-llm-inference-runtime-prepare 'source)
        (let ((before
               (irt--bindings nl-llm-inference-runtime--targets))
              (calls 0)
              (compiler (symbol-function 'byte-compile)))
          (cl-letf (((symbol-function 'byte-compile)
                     (lambda (definition)
                       (setq calls (1+ calls))
                       (if (= calls 3)
                           (error "injected compiler failure")
                         (funcall compiler definition)))))
            (irt--check "auto compiler failure falls back transactionally"
                        (eq (nl-llm-inference-runtime-prepare 'auto) 'source))
            (irt--check "compiler failure occurred after partial compile"
                        (= calls 3) (format "calls=%d" calls))
            (irt--check "compiler failure leaves every source binding intact"
                        (irt--bindings-match-p before))
            (setq calls 0)
            (irt--check "explicit byte-code propagates compiler failure"
                        (condition-case nil
                            (progn
                              (nl-llm-inference-runtime-prepare 'byte-code)
                              nil)
                          (error t)))))

        (cl-letf (((symbol-function 'byte-compile)
                   (lambda (_definition) (signal 'quit nil))))
          (irt--check "auto never swallows compiler quit"
                      (condition-case nil
                          (progn
                            (nl-llm-inference-runtime-prepare 'auto)
                            nil)
                        (quit t))))

        (nl-llm-inference-runtime-prepare 'byte-code)
        (let* ((target 'nl-llm-silu)
               (original (cdr (assq target initial-bindings))))
          (unwind-protect
              (progn
                (fset target 'identity)
                (irt--check "auto rejects symbolic target alias safely"
                            (eq (nl-llm-inference-runtime-prepare 'auto)
                                'source))
                (irt--check "symbolic target alias remains externally owned"
                            (eq (symbol-function target) 'identity)))
            (nl-llm-inference-runtime-prepare 'source)
            (fset target original)))

        ;; A compiler hook can redefine a source after producing byte-code.  The
        ;; precommit check must reject the batch without overwriting that write.
        (nl-llm-inference-runtime-prepare 'source)
        (let* ((target 'nl-llm-silu)
               (original (cdr (assq target initial-bindings)))
               (replacement (lambda (&rest args) (apply original args)))
               (compiler (symbol-function 'byte-compile))
               (calls 0))
          (unwind-protect
              (cl-letf (((symbol-function 'byte-compile)
                         (lambda (definition)
                           (setq calls (1+ calls))
                           (let ((compiled (funcall compiler definition)))
                             (when (= calls
                                      (length
                                       nl-llm-inference-runtime--targets))
                               (fset target replacement))
                             compiled))))
                (irt--check "precommit source change makes auto fall back"
                            (eq (nl-llm-inference-runtime-prepare 'auto)
                                'source))
                (irt--check "precommit preserves external source definition"
                            (eq (symbol-function target) replacement)))
            (nl-llm-inference-runtime-prepare 'source)
            (fset target original)))

        (cl-letf (((symbol-function 'byte-compile) nil))
          (irt--check "auto without compiler falls back to source"
                      (eq (nl-llm-inference-runtime-prepare 'auto) 'source))
          (irt--check "explicit byte-code rejects missing compiler"
                      (condition-case nil
                          (progn
                            (nl-llm-inference-runtime-prepare 'byte-code)
                            nil)
                        (error t))))
        (irt--check "invalid mode is rejected"
                    (condition-case nil
                        (progn
                          (nl-llm-inference-runtime-prepare 'bogus)
                          nil)
                      (error t))))
    (nl-llm-inference-runtime-prepare 'source)
    (dolist (entry initial-bindings)
      (fset (car entry) (cdr entry)))
    (dolist (entry initial-properties)
      (setplist (car entry) (cdr entry)))))

(princ (format "NL-LLM-INFERENCE-RUNTIME %s (%d failures)\n"
               (if (= irt--fail 0) "ALL-PASS" "HAS-FAILURES") irt--fail))
(kill-emacs (if (= irt--fail 0) 0 1))

;;; inference-runtime-test.el ends here
