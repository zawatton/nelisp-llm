;;; evolve-test.el --- test transactional evolution  -*- lexical-binding: t; -*-

;; A candidate evolves away from an isolated champion copy.  Only a measured
;; improvement is promoted; regressions and failed mutations leave the champion
;; untouched.  Shared parameters stay shared across the copy.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/evolve-test.el

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-evolve)

(defvar evolve--fail 0)

(defun evolve--ck (name ok &optional extra)
  (princ (format "%-56s %s  %s\n" name
                 (if ok "PASS"
                   (setq evolve--fail (1+ evolve--fail))
                   "FAIL")
                 (or extra ""))))

(defun evolve--param (value)
  (photon-autograd-const
   (photon-tensor (list 1) (vector (float value)))))

(defun evolve--fitness (model)
  (aref (photon-tensor-data
         (pav-value (plist-get model :fitness)))
        0))

(defun evolve--set-fitness (model value)
  (aset (photon-tensor-data
         (pav-value (plist-get model :fitness)))
        0 (float value)))

;; Copying must isolate the parent while preserving intentional weight tying.
(let* ((shared (evolve--param 1.0))
       (parent (list :fitness shared :tied shared
                     :config (list :name "parent")))
       (child (nl-llm-evolve-copy-model parent)))
  (evolve--ck "copy preserves tied parameters inside the child"
              (eq (plist-get child :fitness) (plist-get child :tied)))
  (evolve--ck "copy does not share parameters with the parent"
              (not (eq (plist-get child :fitness) shared)))
  (evolve--set-fitness child 9.0)
  (evolve--ck "mutating a child leaves its parent unchanged"
              (= (evolve--fitness parent) 1.0)))

(let* ((original (list :fitness (evolve--param 1.0)))
       (state (nl-llm-evolution-new
               original #'evolve--fitness :min-delta 0.10))
       (promoted
        (nl-llm-evolution-step
         state
         (lambda (candidate _state)
           (evolve--set-fitness candidate 2.0))))
       (regressed
        (nl-llm-evolution-step
         state
         (lambda (candidate _state)
           (evolve--set-fitness candidate -5.0))))
       (too-small
        (nl-llm-evolution-step
         state
         (lambda (candidate _state)
           (evolve--set-fitness candidate 2.05))))
       (failed
        (nl-llm-evolution-step
         state
         (lambda (_candidate _state)
           (error "broken self-edit")))))
  (evolve--ck "an independently measured improvement is promoted"
              (and (eq (plist-get promoted :status) 'promoted)
                   (= (nl-llm-evolution-generation state) 1)
                   (= (evolve--fitness
                       (nl-llm-evolution-champion state))
                      2.0)))
  (evolve--ck "the caller's original model remains an immutable parent"
              (= (evolve--fitness original) 1.0))
  (evolve--ck "a regression is rejected and rolled back"
              (and (eq (plist-get regressed :status) 'rejected)
                   (= (evolve--fitness
                       (nl-llm-evolution-champion state))
                      2.0)))
  (evolve--ck "a gain below min-delta is rejected"
              (and (eq (plist-get too-small :status) 'rejected)
                   (= (nl-llm-evolution-champion-score state) 2.0)))
  (evolve--ck "a failed self-edit is contained and recorded"
              (and (eq (plist-get failed :status) 'error)
                   (stringp (plist-get failed :error))
                   (= (evolve--fitness
                       (nl-llm-evolution-champion state))
                      2.0)))
  (evolve--ck "every attempt is auditable"
              (and (= (nl-llm-evolution-attempts state) 4)
                   (= (length (nl-llm-evolution-history state)) 4))))

(let ((bad (list :fitness (evolve--param 1.0))))
  (evolve--ck "non-numeric evaluation is rejected"
              (condition-case nil
                  (progn
                    (nl-llm-evolution-new bad (lambda (_model) 'unknown))
                    nil)
                (error t))))

(let* ((state
        (nl-llm-evolution-new
         (list :fitness (evolve--param 1.0)) #'evolve--fitness
         :generation 7))
       (result
        (nl-llm-evolution-step
         state
         (lambda (candidate _state)
           (evolve--set-fitness candidate 2.0)))))
  (evolve--ck "restored champion generation continues monotonically"
              (and (= (plist-get result :generation-before) 7)
                   (= (plist-get result :generation-after) 8)
                   (= (nl-llm-evolution-generation state) 8))))

(let* ((published nil)
       (state
        (nl-llm-evolution-new
         (list :fitness (evolve--param 1.0)) #'evolve--fitness
         :publish
         (lambda (candidate entry live-state)
           (setq published
                 (list (evolve--fitness candidate)
                       (plist-get entry :generation-after)
                       (nl-llm-evolution-generation live-state))))))
       (result
        (nl-llm-evolution-step
         state
         (lambda (candidate _state)
           (evolve--set-fitness candidate 3.0)))))
  (evolve--ck "publisher runs before champion commit"
              (and (eq (plist-get result :status) 'promoted)
                   (equal published '(3.0 1 0))
                   (equal (plist-get result :publication) published)
                   (= (nl-llm-evolution-generation state) 1))))

(let* ((state
        (nl-llm-evolution-new
         (list :fitness (evolve--param 1.0)) #'evolve--fitness
         :publish (lambda (_candidate _entry _state)
                    (error "artifact store unavailable"))))
       (result
        (nl-llm-evolution-step
         state
         (lambda (candidate _state)
           (evolve--set-fitness candidate 4.0)))))
  (evolve--ck "publish failure cancels promotion transactionally"
              (and (eq (plist-get result :status) 'error)
                   (eq (plist-get result :stage) 'publish)
                   (= (nl-llm-evolution-generation state) 0)
                   (= (evolve--fitness
                       (nl-llm-evolution-champion state))
                      1.0))))

(princ (format "NL-LLM-EVOLVE %s (%d failures)\n"
               (if (= evolve--fail 0) "ALL-PASS" "HAS-FAILURES")
               evolve--fail))
(kill-emacs (if (= evolve--fail 0) 0 1))

;;; evolve-test.el ends here
