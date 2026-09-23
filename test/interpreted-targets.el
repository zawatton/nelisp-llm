;;; interpreted-targets.el --- put the inference targets back in source form  -*- lexical-binding: t; -*-

;; `nl-llm-inference-runtime' compiles the inference path in memory with
;; inlining inhibited, so that a dependency redefined between turns is actually
;; called.  That guarantee is only expressible when its targets reach it
;; interpreted: `byte-compile' hands an existing byte-code object straight
;; back, so a target loaded from a .elc keeps whatever defsubst bodies were
;; inlined into it at file-compile time -- and the module now says so rather
;; than pretending otherwise.
;;
;; Every test of that module therefore has to choose a world.  Skipping on a
;; compiled checkout would delete the module's whole coverage there, so the
;; tests instead check the refusal against whatever state the tree is in and
;; then reload the target files from their .el, which is what this provides.

(require 'cl-lib)
(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'nl-llm-inference-runtime)

(defun interpreted-targets-precompiled ()
  "Return the inference targets currently defined as byte-code."
  (cl-remove-if-not (lambda (symbol)
                      (and (fboundp symbol)
                           (byte-code-function-p (symbol-function symbol))))
                    nl-llm-inference-runtime--targets))

(defun interpreted-targets-reload ()
  "Reload from source every file defining a target or a dependency.
`symbol-file' names the .elc when one was loaded; the .el beside it is what
the in-memory compiler needs to see.  Returns the targets still compiled,
which is nil when the reload did its job."
  (dolist (file (delete-dups
                 (delq nil
                       (mapcar (lambda (symbol) (symbol-file symbol 'defun))
                               (append nl-llm-inference-runtime--targets
                                       nl-llm-inference-runtime--dependencies)))))
    (let ((source (if (string-suffix-p ".elc" file) (substring file 0 -1) file)))
      (when (file-readable-p source)
        (load source nil t t))))
  (interpreted-targets-precompiled))

(provide 'interpreted-targets)
;;; interpreted-targets.el ends here
