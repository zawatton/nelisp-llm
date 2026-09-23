;;; agent-masked-loss-test.el --- completion-only CPU loss tests  -*- lexical-binding: t; -*-
;; Run: emacs -Q --batch -l test/agent-masked-loss-test.el

(setq load-prefer-newer t)
(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'cl-lib)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-autograd)
(require 'nl-llm-agent-improve)

(defvar aml--fail 0)

(defun aml--check (name ok &optional detail)
  (princ (format "%-52s %s%s\n" name
                 (if ok "PASS" (progn (setq aml--fail (1+ aml--fail)) "FAIL"))
                 (if detail (concat "  " detail) ""))))

(defun aml--logits (rows vocab data)
  (photon-autograd-const
   (photon-tensor (list rows vocab) (apply #'vector data))))

(defun aml--scalar (value)
  (aref (photon-tensor-data (pav-value value)) 0))

(defun aml--finite-p (x)
  (and (= x x) (< (abs x) 1.0e308)))

(defun aml--masked-value (data rows vocab targets mask)
  (photon-autograd-reset-tape)
  (aml--scalar
   (nl-llm-ag-masked-softmax-ce
    (aml--logits rows vocab data)
    (copy-sequence targets) (copy-sequence mask))))

;; All active rows agree with the existing CE implementation at ordinary
;; magnitudes, while the new loss remains finite beyond the old log clamp.
(let* ((data '(0.2 -0.3 0.7 1.1 -0.4 0.3))
       (targets [2 0])
       (masked (progn
                 (photon-autograd-reset-tape)
                 (aml--scalar
                  (nl-llm-ag-masked-softmax-ce
                   (aml--logits 2 3 data) targets [1 1]))))
       (legacy (progn
                 (photon-autograd-reset-tape)
                 (aml--scalar
                  (photon-autograd-softmax-ce
                   (aml--logits 2 3 data) targets)))))
  (aml--check "all-ones mask matches legacy CE"
              (< (abs (- masked legacy)) 1.0e-12)
              (format "masked=%.12g legacy=%.12g" masked legacy)))

(let ((loss (aml--masked-value
             '(1000.0 0.0 -1000.0 -1000.0 1000.0 0.0)
             2 3 [2 0] [1 1])))
  (aml--check "extreme logits have finite stable loss"
              (and (aml--finite-p loss) (< (abs (- loss 2000.0)) 1.0e-10))
              (format "loss=%.12g" loss)))

(let ((loss (aml--masked-value '(1.0e20 1.0e20) 1 2 [0] [1])))
  (aml--check "huge equal logits retain log-two loss"
              (< (abs (- loss (log 2.0))) 1.0e-15)
              (format "loss=%.17g" loss)))

;; Mutating caller-owned vectors after the forward pass cannot retarget or
;; remask its backward closure.  The analytic gradient also exercises a
;; non-unit upstream derivative and is compared with finite differences.
(let* ((data '(0.4 -0.2 0.1 -0.7 0.9 0.3 1.2 -0.1 0.0))
       (targets (vector 0 1 2))
       (mask (vector 1 0 1))
       (scale 2.5)
       (eps 1.0e-5)
       logits loss scaled analytic numerical)
  (photon-autograd-reset-tape)
  (setq logits (aml--logits 3 3 data)
        loss (nl-llm-ag-masked-softmax-ce logits targets mask)
        scaled (photon-autograd-scale loss scale))
  (aset targets 0 2)
  (aset targets 2 0)
  (aset mask 0 0)
  (aset mask 1 1)
  (aset mask 2 0)
  (photon-autograd-backward scaled)
  (setq analytic (copy-sequence (photon-tensor-data (pav-grad logits)))
        numerical (make-vector (length data) 0.0))
  (dotimes (i (length data))
    (let* ((plus (copy-sequence data))
           (minus (copy-sequence data)))
      (setcar (nthcdr i plus) (+ (nth i plus) eps))
      (setcar (nthcdr i minus) (- (nth i minus) eps))
      (aset numerical i
            (/ (- (* scale (aml--masked-value plus 3 3 [0 1 2] [1 0 1]))
                  (* scale (aml--masked-value minus 3 3 [0 1 2] [1 0 1])))
               (* 2.0 eps)))))
  (let ((maxerr 0.0))
    (dotimes (i (length analytic))
      (setq maxerr (max maxerr (abs (- (aref analytic i)
                                        (aref numerical i))))))
    (aml--check "active gradient and upstream scale match finite differences"
                (< maxerr 1.0e-5) (format "maxerr=%.3g" maxerr)))
  (aml--check "inactive logit row has exact zero gradient"
              (and (= (aref analytic 3) 0.0)
                   (= (aref analytic 4) 0.0)
                   (= (aref analytic 5) 0.0))))

;; Every malformed request fails before the masked operation records a tape
;; node.  The target of an inactive row is deliberately still validated.
(dolist (case (list
               (list [0] [1 1])
               (list [0 1] '(1 1))
               (list [0 1] [1])
               (list [0 1] [1 2])
               (list [0 1] [1 0.0])
               (list [0 1] [0 0])
               (list [0 3] [1 0])))
  (photon-autograd-reset-tape)
  (let* ((logits (aml--logits 2 3 '(0.0 0.1 0.2 0.3 0.4 0.5)))
         (before photon-autograd--tape)
         (errored nil))
    (condition-case nil
        (nl-llm-ag-masked-softmax-ce logits (car case) (nth 1 case))
      (error (setq errored t)))
    (aml--check (format "invalid request rejected before tape record: %S" case)
                (and errored (eq before photon-autograd--tape)))))

;; Completion-only masking applies at the output loss.  Prefix tokens remain
;; live attention context and therefore receive embedding gradients.
(let* ((model (nl-llm-agent-improve-model 4 4 96 1 1))
       (loss (nl-llm-agent--p5-forward model '(10 11 12) [11 12 13] [0 0 1]))
       (wte-grad (photon-tensor-data (pav-grad (plist-get model :wte))))
       (dim (plist-get model :dim)))
  (photon-autograd-backward loss)
  (let ((prefix-sum 0.0))
    (dolist (token '(10 11))
      (dotimes (j dim)
        (setq prefix-sum (+ prefix-sum (abs (aref wte-grad (+ (* token dim) j)))))))
    (aml--check "P5 masked loss backpropagates through attention prefix"
                (> prefix-sum 1.0e-12)
                (format "prefix-grad-l1=%.3g" prefix-sum))))

;; MASK=nil retains the exact legacy dispatch rather than routing through the
;; new operator with a synthesized all-ones mask.
(let ((model (nl-llm-agent-improve-model 2 2 96 1 1))
      (legacy-calls 0)
      (masked-calls 0)
      (legacy (symbol-function 'photon-autograd-softmax-ce))
      (masked (symbol-function 'nl-llm-ag-masked-softmax-ce)))
  (cl-letf (((symbol-function 'photon-autograd-softmax-ce)
             (lambda (logits targets)
               (setq legacy-calls (1+ legacy-calls))
               (funcall legacy logits targets)))
            ((symbol-function 'nl-llm-ag-masked-softmax-ce)
             (lambda (logits targets mask)
               (setq masked-calls (1+ masked-calls))
               (funcall masked logits targets mask))))
    (nl-llm-agent--p5-forward model '(1 2) [2 3] nil))
  (aml--check "P5 nil mask preserves legacy CE dispatch"
              (and (= legacy-calls 1) (= masked-calls 0))))

(when (> aml--fail 0)
  (error "agent masked loss: %d failure(s)" aml--fail))
(princ "agent masked loss: all checks passed\n")

;;; agent-masked-loss-test.el ends here
