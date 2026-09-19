;;; weights-backward-test.el --- gradient checks for the block's vjps  -*- lexical-binding: t; -*-
;; Run: emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/weights-backward-test.el
;;
;; Doc 08 Phase 3.  Every piece between one linear and the next, checked against
;; finite differences on its own before anything is composed: RMSNorm, the
;; half-split rotation, QK-norm, causal GQA and SwiGLU.
;;
;; Individually first, deliberately.  A composed check is stronger -- a correct
;; gradient through a whole block implies every vjp inside it -- but when it
;; fails it says only "somewhere in here", and these are functions where a sign
;; or a missing term hides comfortably while still producing something that
;; looks like a descent direction.  So each gets its own check, and several get
;; a control showing what the dropped term costs.
;;
;; The driver is the same throughout: pick a fixed random cotangent w, define
;; L = <w, f(x)>, so dL/dx is exactly the vjp applied to w.  Central differences
;; at 1e-6 then give the reference.

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'nl-llm-attn)
(require 'nl-llm-weights-forward)
(require 'nl-llm-weights-backward)

(defvar wbk--fail 0)
(defun wbk--ck (name ok &optional extra)
  (princ (format "%-52s %s  %s\n" name
                 (if ok "PASS" (progn (setq wbk--fail (1+ wbk--fail)) "FAIL"))
                 (or extra ""))))

(defun wbk--vec (n seed &optional scale)
  (let ((v (make-vector n 0.0)) (s (or scale 1.0)))
    (dotimes (i n)
      (aset v i (* s 0.31 (- (mod (+ (* (1+ i) 6151) (* seed 97)) 97) 48))))
    v))

(defun wbk--dot (a b)
  (let ((acc 0.0)) (dotimes (i (length a)) (setq acc (+ acc (* (aref a i) (aref b i))))) acc))

(defun wbk--check-grad (name f analytic x n &optional tol atol)
  "Compare ANALYTIC against central differences of F over X's first N entries.

An entry passes on EITHER a small relative or a small absolute difference, and
that is not a loosening.  A gradient can be legitimately tiny -- a saturated
softmax gives components around 1e-12 -- and a relative comparison against a
central difference at 1e-6 is then measuring rounding, not correctness.  The
first version of this suite judged relatively only and reported the attention
d/dq as wrong at rel 1.4e-03 when the true value was -1.4e-12: the formula was
right and the metric was not.  ATOL defaults to 1e-9."
  (let ((eps 1.0e-6) (worst 0.0) (at nil) (worst-abs 0.0))
    (dotimes (i n)
      (let* ((saved (aref x i))
             (up (progn (aset x i (+ saved eps)) (funcall f)))
             (down (progn (aset x i (- saved eps)) (funcall f))))
        (aset x i saved)
        (let* ((fd (/ (- up down) (* 2.0 eps)))
               (a (aref analytic i))
               (adiff (abs (- fd a)))
               (m (max (abs fd) (abs a) 1.0e-30))
               (rel (/ adiff m)))
          ;; Only a difference that is large both ways counts against us.
          (when (and (> rel worst) (> adiff (or atol 1.0e-9)))
            (setq worst rel at i))
          (when (> adiff worst-abs) (setq worst-abs adiff)))))
    (wbk--ck name (< worst (or tol 1.0e-5))
             (format "worst rel %.2e (abs %.1e) over %d entries (at %S)"
                     worst worst-abs n at))
    worst))

;;; --- RMSNorm --------------------------------------------------------------

(let* ((n 12)
       (x (wbk--vec n 1 0.4))
       (gain (wbk--vec n 2 0.05))
       (w (wbk--vec n 3 0.2))
       (eps 1.0e-6))
  (dotimes (i n) (aset gain i (+ 0.8 (abs (aref gain i)))))
  (cl-labels ((loss () (wbk--dot w (nl-llm-wf--rmsnorm x 0 n gain eps))))
    (let ((dx (nl-llm-wb-rmsnorm-vjp x 0 n gain eps w)))
      (wbk--check-grad "RMSNorm vjp" (lambda () (loss)) dx x n)

      ;; Control: the non-diagonal term is the one that gets dropped.
      (let* ((diag-only (make-vector n 0.0))
             (ss 0.0))
        (dotimes (j n) (setq ss (+ ss (* (aref x j) (aref x j)))))
        (let ((inv (/ 1.0 (sqrt (+ (/ ss (float n)) eps)))))
          (dotimes (j n)
            (aset diag-only j (* (aref gain j) (aref w j) inv))))
        (let ((m 0.0))
          (dotimes (j n)
            (setq m (max m (/ (abs (- (aref diag-only j) (aref dx j)))
                              (max (abs (aref dx j)) 1.0e-9)))))
          (wbk--ck "control: dropping RMSNorm's mean term is wrong"
                   (> m 0.01) (format "rel %.3f if dropped" m)))))))

;;; --- the rotation ---------------------------------------------------------

(dolist (style '(interleaved half))
  (let* ((hd 6) (nheads 2) (n (* nheads hd)) (pos 3) (rbase 1000000.0)
         (x (wbk--vec n 4 0.5))
         (w (wbk--vec n 5 0.3)))
    (cl-labels ((loss ()
                  (let ((v (copy-sequence x)))
                    (nl-llm--rope-heads v 0 nheads hd pos rbase style)
                    (wbk--dot w v))))
      (let ((dx (nl-llm-wb-rope-vjp (copy-sequence w) 0 nheads hd pos rbase style)))
        (wbk--check-grad (format "RoPE vjp (%s)" style)
                         (lambda () (loss)) dx x n)))))

;; A rotation preserves norms, so its vjp must too -- a cheap invariant that a
;; sign error on the sine breaks while the gradient check might still pass at
;; one position.
(let* ((hd 8) (n hd) (w (wbk--vec n 6 0.4)))
  (let* ((before (sqrt (wbk--dot w w)))
         (rotated (nl-llm-wb-rope-vjp (copy-sequence w) 0 1 hd 5 1000000.0 'half))
         (after (sqrt (wbk--dot rotated rotated))))
    (wbk--ck "RoPE vjp preserves the norm"
             (< (/ (abs (- before after)) before) 1.0e-12)
             (format "%.10f vs %.10f" before after))))

;;; --- QK-norm --------------------------------------------------------------

(let* ((hd 6) (nheads 2) (n (* nheads hd))
       (x (wbk--vec n 7 0.4))
       (gain (wbk--vec hd 8 0.05))
       (w (wbk--vec n 9 0.3))
       (eps 1.0e-6))
  (dotimes (i hd) (aset gain i (+ 0.9 (abs (aref gain i)))))
  (cl-labels ((loss ()
                (let ((v (copy-sequence x)))
                  (nl-llm--rmsnorm-heads v 0 nheads hd
                                         (photon-tensor (list hd) gain) eps)
                  (wbk--dot w v))))
    (let ((dx (nl-llm-wb-rmsnorm-heads-vjp x 0 nheads hd gain
                                           (copy-sequence w) eps)))
      (wbk--check-grad "QK-norm vjp" (lambda () (loss)) dx x n))))

;;; --- SwiGLU ---------------------------------------------------------------

(let* ((n 10)
       (g (wbk--vec n 10 0.6))
       (u (wbk--vec n 11 0.5))
       (w (wbk--vec n 12 0.3)))
  (cl-labels ((loss () (wbk--dot w (nl-llm-wf--silu-mul g u n))))
    (let ((grads (nl-llm-wb-silu-mul-vjp g u w n)))
      (wbk--check-grad "SwiGLU vjp, d/dgate" (lambda () (loss))
                       (nth 0 grads) g n)
      (wbk--check-grad "SwiGLU vjp, d/dup" (lambda () (loss))
                       (nth 1 grads) u n))))

;;; --- attention ------------------------------------------------------------

;; Scale matters here more than anywhere else in this suite.  At 0.5 the scores
;; for one head reached 22.4 against -10.1 -- a spread of 32.5, so the softmax
;; saturates and the true d/dq is around 1e-12, which no finite difference can
;; confirm.  At 0.05 the same scores are 0.22 and -0.10 and the gradient is
;; O(1).  Real attention sees the latter, because RMSNorm and QK-norm put it
;; there; the checker now also passes on absolute agreement so a saturated
;; component cannot be judged relatively.
(let* ((seq 3) (heads 2) (kv-heads 1) (hd 4)
       (qdim (* heads hd)) (kvdim (* kv-heads hd))
       (q (wbk--vec (* seq qdim) 13 0.05))
       (k (wbk--vec (* seq kvdim) 14 0.05))
       (v (wbk--vec (* seq kvdim) 15 0.05))
       (w (wbk--vec (* seq qdim) 16 0.3)))
  (cl-labels ((loss ()
                (wbk--dot w (nl-llm-wf--attend q k v seq heads kv-heads hd))))
    (let ((grads (nl-llm-wb-attend-vjp q k v seq heads kv-heads hd w)))
      (wbk--check-grad "attention vjp, d/dq" (lambda () (loss))
                       (nth 0 grads) q (* seq qdim))
      (wbk--check-grad "attention vjp, d/dk" (lambda () (loss))
                       (nth 1 grads) k (* seq kvdim))
      (wbk--check-grad "attention vjp, d/dv" (lambda () (loss))
                       (nth 2 grads) v (* seq kvdim))

      ;; Control: the softmax Jacobian's subtraction.  Without it dq and dk are
      ;; wrong; dv is untouched, which is why checking only dv would pass.
      (let* ((bad (make-vector (* seq qdim) 0.0))
             (scale (/ 1.0 (sqrt (float hd))))
             (grp (/ heads kv-heads)))
        (dotimes (h heads)
          (let ((qc (* h hd)) (kc (* (/ h grp) hd)))
            (dotimes (i seq)
              (let ((p (make-vector (1+ i) 0.0)) (mx -1.0e30) (sm 0.0))
                (dotimes (j (1+ i))
                  (let ((acc 0.0))
                    (dotimes (t0 hd)
                      (setq acc (+ acc (* (aref q (+ (* i qdim) qc t0))
                                          (aref k (+ (* j kvdim) kc t0))))))
                    (aset p j (* acc scale))
                    (setq mx (max mx (aref p j)))))
                (dotimes (j (1+ i))
                  (aset p j (exp (- (aref p j) mx)))
                  (setq sm (+ sm (aref p j))))
                (dotimes (j (1+ i)) (aset p j (/ (aref p j) sm)))
                (dotimes (j (1+ i))
                  (let ((dp 0.0))
                    (dotimes (t0 hd)
                      (setq dp (+ dp (* (aref w (+ (* i qdim) qc t0))
                                        (aref v (+ (* j kvdim) kc t0))))))
                    ;; ds without the - <p, dp> term
                    (let ((ds (* (aref p j) dp scale)))
                      (dotimes (t0 hd)
                        (aset bad (+ (* i qdim) qc t0)
                              (+ (aref bad (+ (* i qdim) qc t0))
                                 (* ds (aref k (+ (* j kvdim) kc t0)))))))))))))
        (let ((m 0.0) (ref (nth 0 grads)))
          (dotimes (i (* seq qdim))
            (setq m (max m (/ (abs (- (aref bad i) (aref ref i)))
                              (max (abs (aref ref i)) 1.0e-9)))))
          (wbk--ck "control: dropping the softmax subtraction is wrong"
                   (> m 0.01) (format "rel %.3f if dropped" m)))))))

(princ (format "\n%s: %d failure(s)\n"
               (if (zerop wbk--fail) "weights-backward OK" "weights-backward")
               wbk--fail))
(when (> wbk--fail 0) (kill-emacs 1))
