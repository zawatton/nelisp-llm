;;; agent-compact-gpu-seed-test.el --- compact masked GPU graph tests -*- lexical-binding: t; -*-
;; Run: emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/agent-compact-gpu-seed-test.el

(setq load-prefer-newer t)
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-agent-improve)
(require 'nl-llm-gpu)
(require 'nl-llm-gpu-ag)

(defvar acgs--fail 0)

(defconst acgs--block-keys
  '(:ln1g :wq :bq :wk :bk :wv :bv :wo :bo
    :ln2g :wg :bg :wu :bu :wd :bd))

(defun acgs--check (name ok &optional detail)
  (princ (format "%-58s %s%s\n" name
                 (if ok "PASS" (progn (setq acgs--fail (1+ acgs--fail)) "FAIL"))
                 (if detail (concat "  " detail) ""))))

(defun acgs--tensor (shape values)
  (photon-tensor shape
                 (if (vectorp values) (copy-sequence values)
                   (apply #'vector values))))

(defun acgs--maxdiff (left right)
  (let ((maximum 0.0))
    (dotimes (index (length left))
      (setq maximum
            (max maximum (abs (- (aref left index) (aref right index))))))
    maximum))

(defun acgs--onehot (targets vocab)
  (let* ((rows (length targets))
         (data (make-vector (* rows vocab) 0.0)))
    (dotimes (row rows)
      (aset data (+ (* row vocab) (aref targets row)) 1.0))
    (photon-tensor (list rows vocab) data)))

(defun acgs--expanded-scales (scales vocab)
  (let* ((rows (length scales))
         (data (make-vector (* rows vocab) 0.0)))
    (dotimes (row rows)
      (dotimes (column vocab)
        (aset data (+ (* row vocab) column) (aref scales row))))
    (photon-tensor (list rows vocab) data)))

(defun acgs--seed-step (indexed)
  "Return updated logits after dense or INDEXED masked CE seeding."
  (let* ((rows 4)
         (vocab 5)
         (initial
          (vector 80.0 -80.0 0.25 -0.5 1.0
                  -12.0 15.0 0.0 3.0 -2.0
                  -40.0 20.0 60.0 -60.0 0.5
                  1.5 -3.0 2.0 8.0 -9.0))
         ;; Repeated target id 2, with the second repeated row inactive.
         (target-values [2 2 4 2])
         (scale-values [2.0 0.0 2.0 0.0])
         (builder (nlga-new))
         (host (acgs--tensor (list rows vocab) initial))
         (logits (nlga-param builder host))
         (targets
          (nlga-const
           builder
           (if indexed
               (acgs--tensor (list rows) target-values)
             (acgs--onehot target-values vocab))))
         (scales
          (nlga-const
           builder
           (if indexed
               (acgs--tensor (list rows) scale-values)
             (acgs--expanded-scales scale-values vocab)))))
    (unwind-protect
        (progn
          (if indexed
              (nlga-seed-ce-idx-masked builder logits targets scales)
            (nlga-seed-ce-masked builder logits targets scales))
          (nlga-finish builder (nlga-scalar builder 0.05))
          (nlga-compile builder)
          (nlga-step builder)
          (nlga-readback builder)
          (cons initial (copy-sequence (photon-tensor-data host))))
      (nlga-free builder))))

(defun acgs--builder-unchanged-p (builder logits slots dispatches backward)
  (and (= (nlga-nslot builder) slots)
       (equal (nlga-disp builder) dispatches)
       (eq (nlga-bwd builder) backward)
       (null (nlga-rt-grad logits))))

(defun acgs--model-values (model)
  (apply
   #'vconcat
   (mapcar
    (lambda (parameter)
      (copy-sequence (photon-tensor-data (pav-value parameter))))
    (nl-llm-agent--p5-params model))))

(defun acgs--model-step (indexed)
  "Run one full tiny model update using dense or INDEXED input embedding."
  (let* ((model (nl-llm-agent-improve-model 2 2 96 1 1))
         (seq 3)
         (dim 2)
         (heads 1)
         (vocab 96)
         (tokens [5 5 7])
         (targets [5 7 8])
         (scales [1.5 0.0 1.5])
         (builder (nlga-new)))
    (cl-labels
        ((parameter (pav)
           (nlga-param builder (pav-value pav)))
         (block (source)
           (let ((result nil))
             (dolist (key acgs--block-keys)
               (setq result
                     (append result
                             (list key (parameter (plist-get source key))))))
             result)))
      (let* ((input
              (nlga-const
               builder
               (if indexed
                   (acgs--tensor (list seq) tokens)
                 (acgs--onehot tokens vocab))))
             (target-rt
              (nlga-const builder (acgs--tensor (list seq) targets)))
             (scale-rt
              (nlga-const builder (acgs--tensor (list seq) scales)))
             (wte (parameter (plist-get model :wte)))
             (blocks (mapcar #'block (plist-get model :blocks)))
             (lnfg (parameter (plist-get model :lnfg)))
             (wh (parameter (plist-get model :wh)))
             (bh (parameter (plist-get model :bh)))
             (tables (nl-llm-gpu-rope-tables seq dim))
             (cosr (nlga-const builder (car tables)))
             (sinr (nlga-const builder (cdr tables)))
             (spos (nlga-scalar builder 1.0))
             (sneg (nlga-scalar builder -1.0))
             (scale (nlga-scalar builder 1.0))
             (mask
              (nlga-const
               builder
               (acgs--tensor
                (list seq seq)
                (vector 0.0 -1.0e30 -1.0e30
                        0.0 0.0 -1.0e30
                        0.0 0.0 0.0))))
             (logits
              (if indexed
                  (nlga-model-idx
                   builder input wte blocks lnfg wh bh heads heads
                   cosr sinr spos sneg scale mask)
                (nlga-model
                 builder input wte blocks lnfg wh bh heads heads
                 cosr sinr spos sneg scale mask))))
        (unwind-protect
            (progn
              (nlga-seed-ce-idx-masked
               builder logits target-rt scale-rt)
              (nlga-finish builder (nlga-scalar builder 0.01))
              (nlga-compile builder)
              (nlga-step builder)
              (nlga-readback builder)
              (acgs--model-values model))
          (nlga-free builder))))))

(unless (nl-llm-gpu-enable)
  (princ "agent compact GPU seed: SKIP (no Vulkan device)\n")
  (kill-emacs 0))

(unwind-protect
    (progn
      (let* ((dense-pair (acgs--seed-step nil))
             (index-pair (acgs--seed-step t))
             (initial (car index-pair))
             (dense (cdr dense-pair))
             (indexed (cdr index-pair))
             (difference (acgs--maxdiff dense indexed)))
        (acgs--check "masked index CE matches dense masked CE"
                     (< difference 1.0e-6)
                     (format "maxdiff=%.3g" difference))
        (acgs--check
         "zero-scaled repeated-target rows remain exactly unchanged"
         (and
          (cl-loop for offset from 5 below 10
                   always (= (aref indexed offset) (aref initial offset)))
          (cl-loop for offset from 15 below 20
                   always (= (aref indexed offset) (aref initial offset)))))
        (acgs--check
         "masked index CE update remains finite for extreme finite logits"
         (cl-loop for value across indexed
                  always (and (= value value) (< (abs value) 1.0e30)))))

      (let* ((builder (nlga-new))
             (logits
              (nlga-param
               builder (acgs--tensor '(2 3) (make-vector 6 0.0))))
             (targets
              (nlga-const builder (acgs--tensor '(2) [0.0 1.0])))
             (scales
              (nlga-const builder (acgs--tensor '(2) [2.0 0.0]))))
        (unwind-protect
            (progn
              (nlga-seed-ce-idx-masked builder logits targets scales)
              (let ((scale-dispatch (car (nlga-disp builder)))
                    (ce-dispatch (cadr (nlga-disp builder))))
                (acgs--check
                 "compact seed composes existing CE and row-scale shaders"
                 (and (eq (car ce-dispatch) 'ce-grad-idx)
                      (equal (nth 2 ce-dispatch) '(2 3))
                      (eq (car scale-dispatch) 'scale-rows)
                      (equal (nth 2 scale-dispatch) '(2 3))
                      (equal (nth 3 scale-dispatch)
                             (nlga--g 6))
                      (= (car (nth 1 scale-dispatch))
                         (nth 2 (nth 1 scale-dispatch)))))))
          (nlga-free builder)))

      ;; Shape errors leave slots, dispatches, backward thunks, and LOGITS grad
      ;; exactly as they were before the attempted seed.
      (let* ((builder (nlga-new))
             (logits
              (nlga-param
               builder (acgs--tensor '(3 4) (make-vector 12 0.0))))
             (target-ok
              (nlga-const builder (acgs--tensor '(3) [0.0 1.0 2.0])))
             (scale-ok
              (nlga-const builder (acgs--tensor '(3) [1.0 1.0 1.0])))
             (target-cols
              (nlga-const builder (acgs--tensor '(3 2) (make-vector 6 0.0))))
             (target-rows
              (nlga-const builder (acgs--tensor '(2) [0.0 1.0])))
             (scale-cols
              (nlga-const builder (acgs--tensor '(3 2) (make-vector 6 1.0))))
             (empty-logits
              (nlga-rt--make
               :slot (nlga-rt-slot logits) :rows 0 :cols 4))
             (slots (nlga-nslot builder))
             (dispatches (copy-tree (nlga-disp builder)))
             (backward (nlga-bwd builder)))
        (unwind-protect
            (dolist (case (list (list logits target-cols scale-ok)
                                (list logits target-rows scale-ok)
                                (list logits target-ok scale-cols)
                                (list empty-logits target-ok scale-ok)))
              (let ((errored nil))
                (condition-case nil
                    (nlga-seed-ce-idx-masked
                     builder (nth 0 case) (nth 1 case) (nth 2 case))
                  (error (setq errored t)))
                (acgs--check
                 "malformed compact seed shape has no graph mutation"
                 (and errored
                      (acgs--builder-unchanged-p
                       builder (nth 0 case) slots dispatches backward)))))
          (nlga-free builder)))

      (let* ((dense (acgs--model-step nil))
             (indexed (acgs--model-step t))
             (difference (acgs--maxdiff dense indexed)))
        (acgs--check "gather model parameter update matches dense model"
                     (< difference 5.0e-5)
                     (format "maxdiff=%.3g" difference))))
  (nl-llm-gpu-disable))

(when (> acgs--fail 0)
  (error "agent compact GPU seed: %d failure(s)" acgs--fail))
(princ "agent compact GPU seed: all checks passed\n")

;;; agent-compact-gpu-seed-test.el ends here
