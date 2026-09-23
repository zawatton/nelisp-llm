;;; weights-block-backward-test.el --- a gradient through a whole block  -*- lexical-binding: t; -*-
;; Run: emacs -Q --batch -l test/weights-block-backward-test.el
;;
;; Doc 08 Phase 3, the composition.  The individual vjps are checked in
;; test/weights-backward-test.el; this one takes a gradient through all of them
;; at once, which is the check that matters and the one that could not have been
;; written first: when it fails it says only "somewhere in here".
;;
;; Two things it establishes that the parts cannot:
;;
;;   the tape      `nl-llm-wb-block-forward' saves what the backward needs and
;;                 must still produce exactly what `nl-llm-wf-block' produces --
;;                 the oracle the whole import is verified against.  A forward
;;                 that drifted while gaining a tape would make every gradient
;;                 below correct for the wrong function.
;;   the wiring    dL/dA and dL/dB for a LoRA on each role in turn, against
;;                 finite differences through the entire block.  A residual
;;                 added twice, a rotation applied forward instead of
;;                 backward, QK-norm fed the post-norm value instead of the
;;                 pre-norm one: all of those survive the unit checks and die
;;                 here.
;;
;; Seq is 2 on purpose.  At seq 1 attention is a single position attending to
;; itself and the cross-position terms of dq/dk are never exercised.

(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'photon-tensor)
(require 'nl-llm-attn)
(require 'nl-llm-weights)
(require 'nl-llm-lora)
(require 'nl-llm-weights-forward)
(require 'nl-llm-weights-lora)
(require 'nl-llm-weights-backward)

(defvar wbb--fail 0)
(defun wbb--ck (name ok &optional extra)
  (princ (format "%-54s %s  %s\n" name
                 (if ok "PASS" (progn (setq wbb--fail (1+ wbb--fail)) "FAIL"))
                 (or extra ""))))

(defun wbb--vec (n seed &optional scale)
  (let ((v (make-vector n 0.0)) (s (or scale 1.0)))
    (dotimes (i n)
      (aset v i (* s 0.23 (- (mod (+ (* (1+ i) 7717) (* seed 131)) 89) 44))))
    v))

(defun wbb--dot (a b)
  (let ((acc 0.0)) (dotimes (i (length a)) (setq acc (+ acc (* (aref a i) (aref b i))))) acc))

;; A small imported-shaped layer, built without a donor: decoupled head width
;; (heads*hd > dim, as Qwen3 has), grouped kv heads, real gains.
(defvar wbb--dim 8)
(defvar wbb--heads 2)
(defvar wbb--kv 1)
(defvar wbb--hd 6)
(defvar wbb--ff 12)

(defun wbb--gain (n seed)
  (let ((v (make-vector n 0.0)))
    (dotimes (i n) (aset v i (+ 0.85 (* 0.03 (mod (+ i seed) 7)))))
    v))

(defun wbb--layer ()
  (let* ((dim wbb--dim) (qdim (* wbb--heads wbb--hd))
         (kvdim (* wbb--kv wbb--hd)) (ff wbb--ff))
    (nl-llm-wf-layer--make
     :wq (nl-llm-weights-lin-quantize (wbb--vec (* qdim dim) 1 0.05) qdim dim)
     :wk (nl-llm-weights-lin-quantize (wbb--vec (* kvdim dim) 2 0.05) kvdim dim)
     :wv (nl-llm-weights-lin-quantize (wbb--vec (* kvdim dim) 3 0.05) kvdim dim)
     :wo (nl-llm-weights-lin-quantize (wbb--vec (* dim qdim) 4 0.05) dim qdim)
     :wg (nl-llm-weights-lin-quantize (wbb--vec (* ff dim) 5 0.05) ff dim)
     :wu (nl-llm-weights-lin-quantize (wbb--vec (* ff dim) 6 0.05) ff dim)
     :wd (nl-llm-weights-lin-quantize (wbb--vec (* dim ff) 7 0.05) dim ff)
     :ln1g (wbb--gain dim 1) :ln2g (wbb--gain dim 2)
     :q-norm (wbb--gain wbb--hd 3) :k-norm (wbb--gain wbb--hd 4))))

(defvar wbb--cfg (list :dim wbb--dim :heads wbb--heads :kv-heads wbb--kv
                       :head-dim wbb--hd :rope-base 1000000.0
                       :rms-eps 1.0e-6))

;;; --- the tape must not change the forward ---------------------------------

(let* ((lay (wbb--layer)) (seq 2)
       (x (wbb--vec (* seq wbb--dim) 10 0.3))
       (oracle (nl-llm-wf-block lay x seq wbb--cfg))
       (taped (nth 0 (nl-llm-wb-block-forward lay x seq wbb--cfg)))
       (m 0.0))
  (dotimes (i (length oracle))
    (setq m (max m (abs (- (aref oracle i) (aref taped i))))))
  (wbb--ck "the taped forward equals nl-llm-wf-block exactly"
           (= m 0.0) (format "maxdiff %.2e over %d" m (length oracle)))

  ;; And the shape under test really is the awkward one.
  (wbb--ck "the test layer has a decoupled head width"
           (> (* wbb--heads wbb--hd) wbb--dim)
           (format "heads*hd = %d > dim = %d"
                   (* wbb--heads wbb--hd) wbb--dim)))

;;; --- a LoRA on each role, through the whole block -------------------------

(let* ((seq 2) (rank 2)
       (x (wbb--vec (* seq wbb--dim) 11 0.3))
       (w (wbb--vec (* seq wbb--dim) 12 0.2))
       (eps 1.0e-6))
  (dolist (role '(:wq :wk :wv :wo :wg :wu :wd))
    (let* ((lay (wbb--layer))
           (lin (nl-llm-wf-layer-lin lay role))
           (rows (nl-llm-weights-lin-rows lin))
           (cols (nl-llm-weights-lin-cols lin))
           (lora (nl-llm-lora-make rows cols rank (* 2 rank) 5))
           (loras (list role lora))
           (a (photon-tensor-data (plist-get lora :a)))
           (b (photon-tensor-data (plist-get lora :b))))
      ;; B starts at zero, which zeroes dA; move it so both are exercised.
      (dotimes (i (length b))
        (aset b i (* 0.04 (aref (wbb--vec (length b) 13) i))))
      (cl-labels ((loss ()
                    (wbb--dot w (nth 0 (nl-llm-wb-block-forward
                                        lay x seq wbb--cfg loras))))
                  (worst (analytic vec)
                    (let ((worst 0.0) (wabs 0.0))
                      (dotimes (i (length vec))
                        (let* ((saved (aref vec i))
                               (up (progn (aset vec i (+ saved eps)) (loss)))
                               (down (progn (aset vec i (- saved eps)) (loss))))
                          (aset vec i saved)
                          (let* ((fd (/ (- up down) (* 2.0 eps)))
                                 (ad (abs (- fd (aref analytic i))))
                                 (m (max (abs fd) (abs (aref analytic i)) 1.0e-30)))
                            (when (> ad wabs) (setq wabs ad))
                            (when (and (> (/ ad m) worst) (> ad 1.0e-9))
                              (setq worst (/ ad m))))))
                      (cons worst wabs))))
        (let* ((fw (nl-llm-wb-block-forward lay x seq wbb--cfg loras))
               (grads (cdr (nl-llm-wb-block-backward
                            lay (nth 1 fw) w seq wbb--cfg loras)))
               (g (plist-get grads role)))
          (if (null g)
              (wbb--ck (format "LoRA on %s produced gradients" role) nil
                       "no gradients returned")
            (let ((ra (worst (plist-get g :da) a))
                  (rb (worst (plist-get g :db) b)))
              (wbb--ck (format "dL/dA through the block, LoRA on %s" role)
                       (< (car ra) 1.0e-4)
                       (format "worst rel %.2e (abs %.1e) over %d"
                               (car ra) (cdr ra) (length a)))
              (wbb--ck (format "dL/dB through the block, LoRA on %s" role)
                       (< (car rb) 1.0e-4)
                       (format "worst rel %.2e (abs %.1e) over %d"
                               (car rb) (cdr rb) (length b))))))))))

;;; --- and the gradient with respect to the block's input -------------------

(let* ((lay (wbb--layer)) (seq 2) (eps 1.0e-6)
       (x (wbb--vec (* seq wbb--dim) 14 0.3))
       (w (wbb--vec (* seq wbb--dim) 15 0.2)))
  (cl-labels ((loss () (wbb--dot w (nth 0 (nl-llm-wb-block-forward
                                           lay x seq wbb--cfg)))))
    (let* ((fw (nl-llm-wb-block-forward lay x seq wbb--cfg))
           (dx (car (nl-llm-wb-block-backward lay (nth 1 fw) w seq wbb--cfg)))
           (worst 0.0) (wabs 0.0))
      (dotimes (i (length x))
        (let* ((saved (aref x i))
               (up (progn (aset x i (+ saved eps)) (loss)))
               (down (progn (aset x i (- saved eps)) (loss))))
          (aset x i saved)
          (let* ((fd (/ (- up down) (* 2.0 eps)))
                 (ad (abs (- fd (aref dx i))))
                 (m (max (abs fd) (abs (aref dx i)) 1.0e-30)))
            (when (> ad wabs) (setq wabs ad))
            (when (and (> (/ ad m) worst) (> ad 1.0e-9))
              (setq worst (/ ad m))))))
      (wbb--ck "dL/dx through the block (all vjps composed)"
               (< worst 1.0e-4)
               (format "worst rel %.2e (abs %.1e) over %d entries"
                       worst wabs (length x))))))

(princ (format "\n%s: %d failure(s)\n"
               (if (zerop wbb--fail) "weights-block-backward OK"
                 "weights-block-backward")
               wbb--fail))
(when (> wbb--fail 0) (kill-emacs 1))
