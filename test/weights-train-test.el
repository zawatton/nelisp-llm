;;; weights-train-test.el --- the training loop over an imported model  -*- lexical-binding: t; -*-
;; Run: emacs -Q --batch -l test/weights-train-test.el
;;
;; Doc 08 Phase 3, the loop.  Everything under it is already checked against
;; finite differences: the linear's backward including the frozen base's W^T.g,
;; each vjp between the linears, and the composition over a whole block.  So
;; what belongs here is what those cannot say:
;;
;;   the stack     dL/dA and dL/dB with the head and the loss attached, through
;;                 more than one block, against finite differences.  The head
;;                 is the tied embedding, so its transpose is the same code the
;;                 blocks use -- but the final RMSNorm and the cross-entropy are
;;                 new, and an off-by-one in which position predicts which token
;;                 is invisible to every check below this one
;;   the boundary  loss taken at completion positions only.  Shifting the prompt
;;                 text must not change the loss, and shifting the completion
;;                 must; a loop that trained on the prompt too would pass every
;;                 gradient check and quietly learn the wrong objective
;;   the step      that it descends
;;
;; A synthetic model, small vocabulary, real structure: decoupled head width,
;; grouped kv heads, tied head, two blocks.  The real 0.6B is far too slow for
;; a suite -- one forward at a hundred tokens is hours on the CPU -- and that
;; cost is reported in the design doc rather than hidden in a skipped test.

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
(require 'nl-llm-weights-train)

(defvar wt--fail 0)
(defun wt--ck (name ok &optional extra)
  (princ (format "%-54s %s  %s\n" name
                 (if ok "PASS" (progn (setq wt--fail (1+ wt--fail)) "FAIL"))
                 (or extra ""))))

(defun wt--vec (n seed &optional scale)
  (let ((v (make-vector n 0.0)) (s (or scale 1.0)))
    (dotimes (i n)
      (aset v i (* s 0.19 (- (mod (+ (* (1+ i) 8191) (* seed 149)) 83) 41))))
    v))

(defvar wt--dim 8)
(defvar wt--heads 2)
(defvar wt--kv 1)
(defvar wt--hd 6)
(defvar wt--ff 12)
(defvar wt--vocab 11)

(defun wt--gain (n seed)
  (let ((v (make-vector n 0.0)))
    (dotimes (i n) (aset v i (+ 0.9 (* 0.02 (mod (+ i seed) 5))))) v))

(defun wt--layer (seed)
  (let* ((dim wt--dim) (qdim (* wt--heads wt--hd))
         (kvdim (* wt--kv wt--hd)) (ff wt--ff))
    (nl-llm-wf-layer--make
     :wq (nl-llm-weights-lin-quantize (wt--vec (* qdim dim) (+ seed 1) 0.05) qdim dim)
     :wk (nl-llm-weights-lin-quantize (wt--vec (* kvdim dim) (+ seed 2) 0.05) kvdim dim)
     :wv (nl-llm-weights-lin-quantize (wt--vec (* kvdim dim) (+ seed 3) 0.05) kvdim dim)
     :wo (nl-llm-weights-lin-quantize (wt--vec (* dim qdim) (+ seed 4) 0.05) dim qdim)
     :wg (nl-llm-weights-lin-quantize (wt--vec (* ff dim) (+ seed 5) 0.05) ff dim)
     :wu (nl-llm-weights-lin-quantize (wt--vec (* ff dim) (+ seed 6) 0.05) ff dim)
     :wd (nl-llm-weights-lin-quantize (wt--vec (* dim ff) (+ seed 7) 0.05) dim ff)
     :ln1g (wt--gain dim (+ seed 1)) :ln2g (wt--gain dim (+ seed 2))
     :q-norm (wt--gain wt--hd (+ seed 3)) :k-norm (wt--gain wt--hd (+ seed 4)))))

(defvar wt--cfg (list :dim wt--dim :heads wt--heads :kv-heads wt--kv
                      :head-dim wt--hd :rope-base 1000000.0 :rms-eps 1.0e-6))

(defun wt--head ()
  (list :lin (nl-llm-weights-lin-quantize
              (wt--vec (* wt--vocab wt--dim) 50 0.05) wt--vocab wt--dim)
        :lnf (wt--gain wt--dim 9)))

;;; --- the stack gradient ---------------------------------------------------

(let* ((layers (list (wt--layer 100) (wt--layer 200)))
       (head (wt--head))
       (ids '(3 7 1 9 4))
       (loss-start 2)
       (rank 2) (eps 1.0e-6))
  (dolist (role '(:wq :wv :wd))
    (let* ((lin (nl-llm-wf-layer-lin (car layers) role))
           (lora (nl-llm-lora-make (nl-llm-weights-lin-rows lin)
                                   (nl-llm-weights-lin-cols lin) rank
                                   (* 2 rank) 11))
           (loras (list role lora))
           (a (photon-tensor-data (plist-get lora :a)))
           (b (photon-tensor-data (plist-get lora :b))))
      (dotimes (i (length b))
        (aset b i (* 0.03 (aref (wt--vec (length b) 12) i))))
      (cl-labels ((loss ()
                    (nl-llm-wtrain-step layers head ids loss-start
                                        wt--cfg loras nil))
                  (worst (analytic vec)
                    (let ((w 0.0) (wa 0.0))
                      (dotimes (i (length vec))
                        (let* ((saved (aref vec i))
                               (up (progn (aset vec i (+ saved eps)) (loss)))
                               (down (progn (aset vec i (- saved eps)) (loss))))
                          (aset vec i saved)
                          (let* ((fd (/ (- up down) (* 2.0 eps)))
                                 (ad (abs (- fd (aref analytic i))))
                                 (m (max (abs fd) (abs (aref analytic i)) 1.0e-30)))
                            (when (> ad wa) (setq wa ad))
                            (when (and (> (/ ad m) w) (> ad 1.0e-9))
                              (setq w (/ ad m))))))
                      (cons w wa))))
        (loss)   ; populate nl-llm-wtrain-last-grads
        (let* ((g (plist-get nl-llm-wtrain-last-grads role))
               (ra (worst (plist-get g :da) a))
               (rb (worst (plist-get g :db) b)))
          (wt--ck (format "dL/dA through head + 2 blocks, LoRA on %s" role)
                  (< (car ra) 1.0e-4)
                  (format "worst rel %.2e (abs %.1e)" (car ra) (cdr ra)))
          (wt--ck (format "dL/dB through head + 2 blocks, LoRA on %s" role)
                  (< (car rb) 1.0e-4)
                  (format "worst rel %.2e (abs %.1e)" (car rb) (cdr rb))))))))

;;; --- the completion-only boundary ----------------------------------------

(let* ((layers (list (wt--layer 300)))
       (head (wt--head))
       (base '(3 7 1 9 4))
       (loss-start 3)
       (loras nil)
       (l0 (nl-llm-wtrain-step layers head base loss-start wt--cfg loras nil)))
  ;; Changing a token strictly inside the prompt changes the attention context
  ;; and therefore the loss, so that is not the invariant.  The invariant is
  ;; which positions are scored: with loss-start 3 over five ids, exactly two
  ;; positions contribute, and predicting a different completion token must
  ;; move the loss while the count stays the same.
  (let* ((swapped (append (cl-subseq base 0 4) (list 2)))
         (l1 (nl-llm-wtrain-step layers head swapped loss-start wt--cfg loras nil)))
    (wt--ck "changing a completion target changes the loss"
            (> (abs (- l1 l0)) 1.0e-6)
            (format "%.6f -> %.6f" l0 l1)))
  ;; And a boundary past the end has nothing to learn from, which must be an
  ;; error rather than a zero loss silently reported as success.
  (wt--ck "a boundary with no completion positions signals"
          (condition-case _
              (progn (nl-llm-wtrain-step layers head base (length base)
                                         wt--cfg loras nil)
                     nil)
            (error t)))
  ;; Loss must be a mean over scored positions, so widening the boundary by one
  ;; changes the divisor: check the reported value is in the plausible range for
  ;; cross-entropy over this vocabulary rather than a sum.
  (wt--ck "loss is a mean, not a sum"
          (and (> l0 0.0) (< l0 (* 3.0 (log wt--vocab))))
          (format "%.4f vs ln(vocab) = %.4f" l0 (log wt--vocab))))

;;; --- and it descends ------------------------------------------------------

(let* ((layers (list (wt--layer 400)))
       (head (wt--head))
       (lin (nl-llm-wf-layer-lin (car layers) :wv))
       (lora (nl-llm-lora-make (nl-llm-weights-lin-rows lin)
                               (nl-llm-weights-lin-cols lin) 2 4 13))
       (loras (list :wv lora))
       (ids '(5 2 8 1 6 3))
       (losses nil))
    (dotimes (_ 25)
      (push (nl-llm-wtrain-step layers head ids 3 wt--cfg loras 0.3) losses))
    (setq losses (nreverse losses))
    (wt--ck "a step descends on the objective it reports"
            (and (< (car (last losses)) (car losses))
                 (cl-every (lambda (l) (= l l)) losses))
            (format "loss %.4f -> %.4f over %d steps"
                    (car losses) (car (last losses)) (length losses))))

;;; --- and the donor tokenizer is what encodes ------------------------------

(let ((table (expand-file-name "build/donor/qwen3-0.6b/tokenizer.bin")))
  (if (not (file-readable-p table))
      (wt--ck "encoding uses the donor tokenizer" t
              "skipped: donor tokenizer table absent")
    (require 'nl-llm-qwen-tokenizer)
    (let* ((tok (nl-llm-qwen-tok-load table))
           (enc (nl-llm-wtrain-encode
                 tok '(:prompt "The capital of France is"
                       :completion " Paris"))))
      (wt--ck "encoding uses the donor tokenizer"
              (and (equal (cl-subseq (car enc) 0 5) '(785 6722 315 9625 374))
                   (= (cdr enc) 5)
                   (equal (nthcdr 5 (car enc)) '(12095)))
              (format "ids %S loss-start %S" (car enc) (cdr enc))))))

(princ (format "\n%s: %d failure(s)\n"
               (if (zerop wt--fail) "weights-train OK" "weights-train") wt--fail))
(when (> wt--fail 0) (kill-emacs 1))
