;;; nl-llm-stack-paths.el --- find this repo's lisp and the sibling substrate  -*- lexical-binding: t; -*-

;; Every test and example used to carry two lines of its own:
;;
;;   (add-to-list 'load-path (expand-file-name "lisp"))
;;   (add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
;;
;; 132 copies of the first and 119 of the second, and both resolve against
;; `default-directory', so a file only worked when the process happened to
;; start in the repository root.  This resolves the same directories once,
;; from the location of this file rather than from the working directory, so
;; a test can be run from anywhere.
;;
;; Resolution order per dependency, first existing wins:
;;
;;   1. its environment variable (NELISP_PHOTON_LISP / NELISP_GPU_LISP),
;;      which is how a Makefile or CI pins a specific checkout;
;;   2. `vendor/<repo>/lisp' inside this repository, which is where a
;;      submodule or a vendored copy would land;
;;   3. the sibling checkout `../<repo>/lisp', which is the working layout.
;;
;; Keeping 2 above 3 means moving to a submodule later needs no code change.
;; Nothing is required at load time: a missing nelisp-gpu is normal on a
;; machine without the GPU backend, and the CPU path must still load.

;;; Code:

(defun nl-llm-stack-paths--root-from (dir)
  "Walk up from DIR to the checkout that holds a `lisp' directory.
Written as a plain loop rather than with `locate-dominating-file',
which the NeLisp standalone reader does not provide."
  (let ((cur (directory-file-name (expand-file-name dir)))
        (found nil)
        (climbing t))
    (while (and climbing (not found))
      (if (file-directory-p (expand-file-name "lisp" cur))
          (setq found cur)
        (let ((up (directory-file-name (file-name-directory cur))))
          (if (equal up cur)
              (setq climbing nil)
            (setq cur up)))))
    (or found (directory-file-name (expand-file-name ".." dir)))))

(defconst nl-llm-stack-paths-root
  (nl-llm-stack-paths--root-from
   (file-name-directory (or load-file-name buffer-file-name default-directory)))
  "Absolute path of the nelisp-llm checkout this file belongs to.")

(defconst nl-llm-stack-paths-dependencies
  '(("nelisp-photon" . "NELISP_PHOTON_LISP")
    ("nelisp-gpu"    . "NELISP_GPU_LISP"))
  "Sibling repositories whose `lisp' directory this repo loads from.
Each entry is (REPO-NAME . ENVIRONMENT-VARIABLE).")

(defun nl-llm-stack-paths--candidates (repo env-var)
  "Return the directories to try for REPO, in order.
ENV-VAR is consulted first so a caller can pin a checkout."
  (delq nil
        (list (let ((v (getenv env-var)))
                (and v (not (string-empty-p v)) (expand-file-name v)))
              (expand-file-name (concat "vendor/" repo "/lisp")
                                nl-llm-stack-paths-root)
              (expand-file-name (concat "../" repo "/lisp")
                                nl-llm-stack-paths-root))))

(defun nl-llm-stack-paths-locate (repo &optional env-var)
  "Return REPO's `lisp' directory, or nil when no candidate exists.
ENV-VAR defaults to the one registered in
`nl-llm-stack-paths-dependencies'."
  (let ((env (or env-var (cdr (assoc repo nl-llm-stack-paths-dependencies)))))
    (seq-find #'file-directory-p (nl-llm-stack-paths--candidates repo env))))

;;;###autoload
(defun nl-llm-stack-paths-ensure ()
  "Put this repo's `lisp' and every dependency found on `load-path'.
Returns the directories that were added or already present.  Missing
dependencies are skipped silently; the caller's own `require' is what
reports a genuinely absent one, with the name of what it wanted."
  (let ((dirs (list (expand-file-name "lisp" nl-llm-stack-paths-root))))
    (dolist (dep nl-llm-stack-paths-dependencies)
      (let ((dir (nl-llm-stack-paths-locate (car dep) (cdr dep))))
        (when dir (push dir dirs))))
    (setq dirs (nreverse dirs))
    (dolist (dir dirs) (add-to-list 'load-path dir))
    dirs))

(nl-llm-stack-paths-ensure)

(provide 'nl-llm-stack-paths)
;;; nl-llm-stack-paths.el ends here
