;;; nl-llm-deltanet-gpu.el --- the gated delta rule on the device -*- lexical-binding: t; -*-

;; The recurrence is where a DeltaNet block's time goes: 48 heads, a 128x128
;; state each, updated at every position, all in Elisp.  Measured on Ternary
;; Bonsai that is about twelve of a block's fourteen seconds, and the
;; projections around it -- already on the GPU -- account for most of the rest.
;;
;; It is sequential in position and parallel in everything else.  For a fixed
;; head and output dimension the whole update touches one column of that head's
;; state, so a thread owns a column, and the only thing that has to be ordered
;; is the positions.  That makes it one dispatch per position in a single
;; batch, with the state living in a tmp slot across them -- no round trip per
;; step, and nothing copied back until the end.

;;; Code:

(require 'nl-llm-deltanet)
(require 'nl-llm-gpu)            ; puts the nelisp-gpu sibling dir on load-path
(require 'nelisp-gpu-server)

;;;###autoload
(defun nl-llm-dngpu-available-p ()
  "Non-nil when the GPU server is up and knows the recurrence kernel."
  (and (fboundp 'nelisp-gpu-server-batch)
       (assq 'gdn-step nelisp-gpu-kernels)))

;;;###autoload
(defun nl-llm-dngpu-scan (qn kn v gates betas seq nv dk dv)
  "Run the gated delta rule for all NV heads on the GPU; return SEQ x NV x DV.

QN and KN are the queries and keys AFTER the L2 norm, laid out
[position][head][dk]; V is [position][head][dv]; GATES and BETAS are
[position][head].  Everything the recurrence needs is already a number here --
the normalisation, the decay and the write strength are cheap and stay on the
host, and what moves is the part that is not."
  (let ((slots (list (cons 'tmp (* nv dk dv))
                     (cons 'in qn) (cons 'in kn) (cons 'in v)
                     (cons 'in gates) (cons 'in betas)
                     (cons 'out (* seq nv dv))))
        (disps nil))
    (dotimes (tt seq)
      (push (list 'gdn-step '(0 1 2 3 4 5 6)
                  (list nv dk dv tt)
                  (/ (+ (* nv dv) 63) 64))
            disps))
    (car (nelisp-gpu-server-batch slots (nreverse disps)))))

;;;###autoload
(defun nl-llm-dngpu-prepare (q k a b alog dtb seq nv dk)
  "Normalise Q and K and evaluate the gates, in the layout `nl-llm-dngpu-scan' wants.
Q and K are [position][head][dk] before normalisation, A and B
[position][head].  Returns (QN KN GATES BETAS)."
  (let ((qn (make-vector (* seq nv dk) 0.0))
        (kn (make-vector (* seq nv dk) 0.0))
        (gates (make-vector (* seq nv) 0.0))
        (betas (make-vector (* seq nv) 0.0))
        (isq (/ 1.0 (sqrt (float dk)))))
    (dotimes (tt seq)
      (dotimes (h nv)
        (let* ((base (* (+ (* tt nv) h) dk))
               (qs (nl-llm-dn--scaled q base dk isq))
               (ks (nl-llm-dn--scaled k base dk isq))
               (qq (if nl-llm-dn-scale-after-norm
                       (nl-llm-dn--scaled
                        (nl-llm-dn--l2norm (nl-llm-dn--scaled q base dk 1.0) dk)
                        0 dk isq)
                     (nl-llm-dn--l2norm qs dk)))
               (kk (if nl-llm-dn-scale-after-norm
                       (nl-llm-dn--l2norm (nl-llm-dn--scaled k base dk 1.0) dk)
                     (nl-llm-dn--l2norm ks dk))))
          (dotimes (i dk)
            (aset qn (+ base i) (aref qq i))
            (aset kn (+ base i) (aref kk i)))
          (aset gates (+ (* tt nv) h)
                (nl-llm-dn-gate (aref a (+ (* tt nv) h))
                                (aref alog h) (aref dtb h)))
          (aset betas (+ (* tt nv) h)
                (nl-llm-dn-beta (aref b (+ (* tt nv) h)))))))
    (list qn kn gates betas)))

(provide 'nl-llm-deltanet-gpu)
;;; nl-llm-deltanet-gpu.el ends here
