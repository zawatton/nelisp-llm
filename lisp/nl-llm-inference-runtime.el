;;; nl-llm-inference-runtime.el --- in-memory CPU inference compiler  -*- lexical-binding: t; -*-

;; The standalone NeLisp runtime executes this model from source.  Emacs can
;; make the same pure-Elisp numeric path substantially faster by byte-compiling
;; a small, measured allowlist in memory.  No .elc or native-comp artifact is
;; created, so the model remains self-extensible and NeLisp-compatible.

;;; Code:

(require 'cl-lib)
(require 'nl-llm-decode)

(defvar macroexp-inhibit-compiler-macros)
(defvar byte-optimize)

(defgroup nl-llm-inference-runtime nil
  "Runtime selection for pure-Elisp nelisp-llm inference."
  :group 'applications)

(defcustom nl-llm-inference-runtime-mode 'auto
  "How `nl-llm-inference-runtime-prepare' prepares CPU inference.
`auto' uses in-memory byte-code when Emacs exposes `byte-compile', and source
otherwise.  `source' restores runtime-owned bindings.  `byte-code' requires the
compiler and signals an error when compilation cannot complete transactionally."
  :type '(choice (const :tag "Automatic" auto)
                 (const :tag "Source" source)
                 (const :tag "In-memory byte-code" byte-code))
  :group 'nl-llm-inference-runtime)

(defconst nl-llm-inference-runtime--targets
  '(photon-tensor-linear
    photon-tensor-matmul
    photon-tensor-softmax-rows
    photon-tensor-scale
    photon-tensor-add
    photon-tensor-hadamard
    nl-llm-rmsnorm
    nl-llm-silu
    nl-llm--rope-block
    nl-llm--rope-heads
    nl-llm--swiglu-b
    nl-llm-decode-block
    nl-llm-decode-step)
  "Measured numeric functions compiled by the Emacs inference adapter.")

(defconst nl-llm-inference-runtime--dependencies
  '(photon-tensor
    photon-tensor-shape
    photon-tensor-data
    nl-llm-dcache-k
    nl-llm-dcache-v
    nl-llm-dcache-len
    nl-llm-dcache-kvdim
    nl-llm-dcache-dim
    nl-llm-dcache-heads
    nl-llm-dcache-kvh
    nl-llm-dcache-k--inliner
    nl-llm-dcache-v--inliner
    nl-llm-dcache-len--inliner
    nl-llm-dcache-kvdim--inliner
    nl-llm-dcache-dim--inliner
    nl-llm-dcache-heads--inliner
    nl-llm-dcache-kvh--inliner)
  "Function and inliner dependencies embedded by compilation of the targets.")

(defvar nl-llm-inference-runtime--owned nil
  "Alist of target symbols and byte-code definitions installed by this module.")
(defvar nl-llm-inference-runtime--sources nil
  "Alist of target symbols and definitions replaced by the owned definitions.")
(defvar nl-llm-inference-runtime--context nil
  "Compilation-context fingerprint associated with the owned definitions.")

(defun nl-llm-inference-runtime--function (symbol)
  "Return SYMBOL's function definition, or signal a useful error."
  (unless (fboundp symbol)
    (error "Inference runtime dependency is unavailable: %s" symbol))
  (symbol-function symbol))

(defun nl-llm-inference-runtime--properties (symbol)
  "Return an isolated snapshot of SYMBOL's compilation metadata."
  ;; Full plists are small for this finite set and include defsubst
  ;; byte-optimizer, compiler-macro, and generalized-variable metadata.  Using
  ;; `copy-tree' also detects in-place edits to property values.
  (copy-tree (symbol-plist symbol)))

(defun nl-llm-inference-runtime--capture-context ()
  "Capture definitions and properties that can affect target compilation."
  (list
   (mapcar
    (lambda (symbol)
      (list symbol
            (nl-llm-inference-runtime--function symbol)
            (nl-llm-inference-runtime--properties symbol)))
    nl-llm-inference-runtime--dependencies)
   (mapcar
    (lambda (symbol)
      (cons symbol (nl-llm-inference-runtime--properties symbol)))
    nl-llm-inference-runtime--targets)))

(defun nl-llm-inference-runtime--context-current-p (context)
  "Return non-nil when current compilation metadata still matches CONTEXT."
  (and context
       (cl-every
        (lambda (entry)
          (let ((symbol (nth 0 entry)))
            (and (fboundp symbol)
                 (eq (symbol-function symbol) (nth 1 entry))
                 (equal (symbol-plist symbol) (nth 2 entry)))))
        (car context))
       (cl-every
        (lambda (entry)
          (equal (symbol-plist (car entry)) (cdr entry)))
        (cadr context))))

(defun nl-llm-inference-runtime--owned-current-p ()
  "Return non-nil when every installed target is still owned by this module."
  (and (= (length nl-llm-inference-runtime--owned)
          (length nl-llm-inference-runtime--targets))
       (cl-every
        (lambda (entry)
          (and (fboundp (car entry))
               (eq (symbol-function (car entry)) (cdr entry))))
        nl-llm-inference-runtime--owned)))

(defun nl-llm-inference-runtime--restore-owned ()
  "Restore only definitions which are still installed and owned by this module.
An external redefinition is deliberately retained.  Runtime ownership records
are cleared even when one or more bindings were superseded externally."
  (let ((owned nl-llm-inference-runtime--owned)
        (sources nl-llm-inference-runtime--sources))
    (setq nl-llm-inference-runtime--owned nil
          nl-llm-inference-runtime--sources nil
          nl-llm-inference-runtime--context nil)
    (dolist (entry owned)
      (let* ((symbol (car entry))
             (installed (cdr entry))
             (source (assq symbol sources)))
        (when (and source (fboundp symbol)
                   (eq (symbol-function symbol) installed))
          (fset symbol (cdr source)))))))

(defun nl-llm-inference-runtime--sources-current-p (sources context)
  "Return non-nil when SOURCES and CONTEXT still match before installation."
  (and
   (cl-every
    (lambda (entry)
      (and (fboundp (car entry))
           (eq (symbol-function (car entry)) (cdr entry))))
    sources)
   (nl-llm-inference-runtime--context-current-p context)))

(defun nl-llm-inference-runtime--captured-closure-p (definition)
  "Return non-nil when DEFINITION captures mutable lexical state.

The byte compiler gives interpreted functions an environment in slot 2.  A
plain named function has the sentinel environment `(t)'; a lexical closure
with captured bindings has additional entries.  The older list-shaped
`closure' representation is retained as a defensive fallback for runtimes
which do not expose `interpreted-function-p'."
  (cl-labels
      ((captured-environment-p (environment)
         ;; Compiler environments also contain non-binding symbols such as
         ;; `cl-struct-...-tags'.  Only dotted binding entries represent
         ;; captured mutable values; the sentinel `t' is harmless metadata.
         (let ((tail environment) found)
           (while (and (not found) (consp tail))
             (setq found (consp (car tail))
                   tail (cdr tail)))
           found)))
    (cond
   ((and (fboundp 'interpreted-function-p)
         (interpreted-function-p definition))
    (let ((environment (aref definition 2)))
      (captured-environment-p environment)))
   ((and (consp definition) (eq (car definition) 'closure))
    (let ((environment (cadr definition)))
      (captured-environment-p environment)))
   (t nil))))

(defun nl-llm-inference-runtime--compile-group ()
  "Compile and install the target group transactionally.
The compiler must already be available.  Definitions are compiled before any
target is changed.  A compiler hook which changes a captured source or inline
dependency makes the transaction fail rather than overwrite that change."
  (let* ((sources
          (mapcar
           (lambda (symbol)
             (cons symbol (nl-llm-inference-runtime--function symbol)))
           nl-llm-inference-runtime--targets))
         (context (nl-llm-inference-runtime--capture-context))
         (compiled
          ;; A cl-defstruct getter keeps its original compiler-macro property
          ;; after an external `fset'.  Do not embed that stale accessor body;
          ;; dynamic calls preserve the redefinition semantics.  Defsubst's
          ;; byte optimizer remains tracked separately through symbol metadata.
          (let ((macroexp-inhibit-compiler-macros t)
                ;; In particular, prevent `byte-compile-inline-expand' from
                ;; compiling a source defsubst into its global function cell as
                ;; a side effect.  The measured gain comes primarily from the
                ;; decoder loops, and dependency calls remain dynamically safe.
                (byte-optimize nil))
            (mapcar
             (lambda (entry)
               (when (nl-llm-inference-runtime--captured-closure-p
                      (cdr entry))
                 (error "Inference target %s captures mutable lexical state"
                        (car entry)))
               ;; `byte-compile' treats a symbol definition as an alias target
               ;; and may compile that other symbol globally.  Such aliases are
               ;; outside this bounded adapter; source mode preserves them.
               (when (symbolp (cdr entry))
                 (error "Inference target %s is an alias to %s"
                        (car entry) (cdr entry)))
               (let ((result (byte-compile (cdr entry))))
                 (unless (byte-code-function-p result)
                   (error "Compiler did not produce byte-code for %s"
                          (car entry)))
                 (cons (car entry) result)))
             sources))))
    (unless (nl-llm-inference-runtime--sources-current-p sources context)
      (error "Inference definitions changed during compilation"))
    (let ((installed nil))
      (condition-case err
          (let ((inhibit-quit t))
            (dolist (entry compiled)
              ;; Record the candidate first.  If `fset' itself signals, the
              ;; rollback equality guard distinguishes no-write from a write.
              (push entry installed)
              (fset (car entry) (cdr entry)))
            (setq nl-llm-inference-runtime--sources sources
                  nl-llm-inference-runtime--owned compiled
                  nl-llm-inference-runtime--context context))
        ((error quit)
         ;; Roll back only bindings successfully installed by this transaction.
         ;; Arbitrary compiler or hook writes are not claimed as controllable.
         (dolist (entry installed)
           (let ((source (assq (car entry) sources)))
             (when (and source (fboundp (car entry))
                        (eq (symbol-function (car entry)) (cdr entry)))
               (fset (car entry) (cdr source)))))
         (signal (car err) (cdr err)))))))

(defun nl-llm-inference-runtime--prepare-byte-code ()
  "Prepare owned byte-code, refreshing stale targets or dependencies."
  (if (and (nl-llm-inference-runtime--owned-current-p)
           (nl-llm-inference-runtime--context-current-p
            nl-llm-inference-runtime--context))
      'byte-code
    ;; Restore the subset still owned.  Externally redefined targets survive and
    ;; become the new source snapshot for the complete-group refresh.
    (nl-llm-inference-runtime--restore-owned)
    ;; Availability was checked before requiring the compiler.  This distinction
    ;; matters on NeLisp, where byte-code objects exist but `byte-compile' does not.
    (require 'bytecomp)
    (nl-llm-inference-runtime--compile-group)
    'byte-code))

;;;###autoload
(defun nl-llm-inference-runtime-prepare (&optional mode)
  "Prepare the pure-Elisp inference path according to MODE.
MODE is `auto', `source', or `byte-code', and defaults to
`nl-llm-inference-runtime-mode'.  Return the selected effective mode, either
`source' or `byte-code'.

Compilation is entirely in memory.  Repeated calls reuse byte-code only while
all installed bindings remain owned and every tracked function/property used by
inline expansion is unchanged.  Otherwise the complete group is refreshed.
Call this before each inference invocation to observe redefinitions between
turns; changes during an invocation are outside this lifecycle boundary.

`source' and failed `auto' preparation restore only bindings still owned by this
module.  `auto' falls back to source when the compiler is absent or compilation
fails.  Symbolic aliases are deliberately unsupported because compiling one can
mutate its alias target.  Explicit `byte-code' signals instead."
  (let ((requested (or mode nl-llm-inference-runtime-mode)))
    (unless (memq requested '(auto source byte-code))
      (error "Invalid inference runtime mode: %S" requested))
    (cond
     ((eq requested 'source)
      (nl-llm-inference-runtime--restore-owned)
      'source)
     ;; Check before `require': NeLisp reports byte-code functions but has no
     ;; compiler, whereas Emacs normally exposes `byte-compile' as an autoload.
     ((not (fboundp 'byte-compile))
      (nl-llm-inference-runtime--restore-owned)
      (if (eq requested 'auto)
          'source
        (error "In-memory byte compiler is unavailable")))
     (t
     (condition-case err
          (nl-llm-inference-runtime--prepare-byte-code)
        (quit
         (nl-llm-inference-runtime--restore-owned)
         (signal (car err) (cdr err)))
        (error
         (nl-llm-inference-runtime--restore-owned)
         (if (eq requested 'auto)
             'source
           (signal (car err) (cdr err)))))))))

(provide 'nl-llm-inference-runtime)
;;; nl-llm-inference-runtime.el ends here
