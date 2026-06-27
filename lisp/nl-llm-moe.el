;;; nl-llm-moe.el --- sparse mixture-of-experts FFN  -*- lexical-binding: t; -*-

;; Top-k sparse Mixture of Experts: a router linear picks the K highest-scoring
;; experts per token, their softmax gate weights are renormalised over the K,
;; and the token output is the gated sum of those experts' SwiGLU FFNs.  Each
;; expert is a SwiGLU block (nl-llm-arch).  Built on photon-tensor.

;;; Code:

(require 'photon-tensor)
(require 'nl-llm-arch)

(defun nl-llm--topk-indices (vec k)
  "Return the indices of the K largest elements of float vector VEC."
  (let ((pairs nil) (n (length vec)) (i 0))
    (while (< i n) (push (cons (aref vec i) i) pairs) (setq i (1+ i)))
    (setq pairs (sort pairs (lambda (a b) (> (car a) (car b)))))
    (let ((out nil) (c 0))
      (dolist (p pairs) (when (< c k) (push (cdr p) out) (setq c (1+ c))))
      (nreverse out))))

;;;###autoload
(defun nl-llm-moe (x router experts top-k)
  "Top-K sparse mixture-of-experts over X (seq x dim).
ROUTER is (E x dim) routing weights; EXPERTS is a list of E plists, each with
:wg :wu (ff x dim) and :wd (dim x ff) for a SwiGLU expert.  Returns (seq x dim)."
  (let* ((sh (photon-tensor-shape x)) (seq (car sh)) (dim (nth 1 sh))
         (ne (length experts))
         (logits (photon-tensor-data (photon-tensor-linear x router)))  ; seq x E
         (gate (make-vector (* seq ne) 0.0))
         (out (make-vector (* seq dim) 0.0)))
    ;; per-row gating weights: softmax over E, keep top-k, renormalise over them
    (dotimes (i seq)
      (let ((base (* i ne)) (mx -1.0e30) (probs (make-vector ne 0.0)) (sm 0.0))
        (dotimes (e ne) (when (> (aref logits (+ base e)) mx) (setq mx (aref logits (+ base e)))))
        (dotimes (e ne)
          (let ((p (exp (- (aref logits (+ base e)) mx)))) (aset probs e p) (setq sm (+ sm p))))
        (dotimes (e ne) (aset probs e (/ (aref probs e) sm)))
        (let ((idxs (nl-llm--topk-indices probs top-k)) (tsum 0.0))
          (dolist (e idxs) (setq tsum (+ tsum (aref probs e))))
          (dolist (e idxs) (aset gate (+ base e) (/ (aref probs e) tsum))))))
    ;; accumulate gated expert outputs (dense compute, sparse gate)
    (dotimes (e ne)
      (let ((ex (nth e experts)))
        (when (let ((any nil) (i 0))
                (while (and (not any) (< i seq))
                  (when (/= (aref gate (+ (* i ne) e)) 0.0) (setq any t)) (setq i (1+ i)))
                any)
          (let ((ye (photon-tensor-data
                     (nl-llm-swiglu x (plist-get ex :wg) (plist-get ex :wu) (plist-get ex :wd)))))
            (dotimes (i seq)
              (let ((w (aref gate (+ (* i ne) e))))
                (unless (= w 0.0)
                  (let ((t0 0))
                    (while (< t0 dim)
                      (aset out (+ (* i dim) t0)
                            (+ (aref out (+ (* i dim) t0)) (* w (aref ye (+ (* i dim) t0)))))
                      (setq t0 (1+ t0)))))))))))
    (photon-tensor (list seq dim) out)))

(provide 'nl-llm-moe)
;;; nl-llm-moe.el ends here
