;;; nl-llm-agent-ondevice.el --- close the self-improvement loop on the GPU  -*- lexical-binding: t; -*-

;; The final piece of the agent harness (docs/design/05-agent-harness.org): run the
;; WHOLE self-improvement loop on-device by transferring the GPU-trained weights
;; back into the rollout decoder.
;;
;; One model, two views that SHARE the same weight tensors:
;;   * a CPU view (`nl-llm-agent-improve-model', stacked nl-llm-ag-block) used to
;;     ROLLOUT actions; and
;;   * an nlga GPU graph (`nlga-model') built FROM the very same host tensors, used
;;     to TRAIN.
;; The two are the same function (verified: CPU vs nlga logits agree to ~2e-5), so
;; after `nlga-step' trains on the GPU, `nlga-readback' copies the trained weights
;; straight back into the shared host tensors -- the CPU rollout instantly decodes
;; with the GPU-trained weights.  No reformatting, no second copy: the transfer is
;; the readback into the shared tensor objects.
;;
;; Loop: rollout (CPU) -> keep reward>=1 -> train (GPU) -> readback -> repeat.  The
;; rollout success rate rising IS the proof the transfer works -- the only thing
;; that changed the CPU decoder's weights is the GPU training fed back through it.

;;; Code:

(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-gpu)
(require 'nl-llm-gpu-ag)
(require 'nl-llm-agent-model)
(require 'nl-llm-agent-improve)

(defun nl-llm-agent--onehot-pad (toks seq vocab)
  "Onehot (SEQ x VOCAB) of TOKS (a list/vector of ids), padded with id 0."
  (let ((v (make-vector (* seq vocab) 0.0)) (i 0) (cs (append toks nil)))
    (while (and cs (< i seq)) (aset v (+ (* i vocab) (car cs)) 1.0) (setq cs (cdr cs) i (1+ i)))
    (while (< i seq) (aset v (+ (* i vocab) 0) 1.0) (setq i (1+ i)))
    (photon-tensor (list seq vocab) v)))

(defun nl-llm-agent--shift-pad (toks seq)
  "Next-token targets of TOKS padded to SEQ (id 0 past the end)."
  (let ((tv (make-vector seq 0)) (a (apply #'vector (append toks nil))) (n 0))
    (setq n (length a))
    (dotimes (i seq) (aset tv i (if (< (1+ i) n) (aref a (1+ i)) 0)))
    tv))

;;;###autoload
(defun nl-llm-agent-ondevice-new (dim ff heads nblocks seq lr)
  "Build a CPU rollout model + an nlga GPU training graph that SHARE weight tensors
\(MHA, kv-heads = HEADS).  SEQ is the fixed training sequence length, LR the Adam
step.  Returns a context plist; free with `nl-llm-agent-ondevice-free'.  Requires
an active GPU (`nl-llm-gpu-enable')."
  (let* ((vocab nl-llm-agent-char-vocab) (hd (/ dim heads)) (scl (/ 1.0 (sqrt (float hd))))
         (cpu (nl-llm-agent-improve-model dim ff vocab nblocks heads))
         (tables (nl-llm-gpu-rope-tables seq hd))
         (mask (let ((md (make-vector (* seq seq) 0.0)) (i 0))
                 (while (< i seq) (let ((j (1+ i))) (while (< j seq) (aset md (+ (* i seq) j) -1.0e30) (setq j (1+ j)))) (setq i (1+ i)))
                 (photon-tensor (list seq seq) md)))
         (b (nlga-new)))
    (cl-flet ((wp (pav) (nlga-param b (pav-value pav)))
              (wb (blk) (let ((out nil) (kv blk)) (while kv (push (car kv) out) (push (nlga-param b (pav-value (cadr kv))) out) (setq kv (cddr kv))) (nreverse out))))
      (let* ((oh (nlga-const b (photon-tensor (list seq vocab) (make-vector (* seq vocab) 0.0))))
             (ohtgt (nlga-const b (photon-tensor (list seq vocab) (make-vector (* seq vocab) 0.0))))
             (wter (wp (plist-get cpu :wte)))
             (blks (mapcar (lambda (blk) (wb blk)) (plist-get cpu :blocks)))
             (lnfgr (wp (plist-get cpu :lnfg))) (whr (wp (plist-get cpu :wh))) (bhr (wp (plist-get cpu :bh)))
             (cosr (nlga-const b (car tables))) (sinr (nlga-const b (cdr tables)))
             (sposr (nlga-scalar b 1.0)) (snegr (nlga-scalar b -1.0)) (sclr (nlga-scalar b scl)) (oner (nlga-scalar b 1.0))
             (maskr (nlga-const b mask))
             (logits (nlga-model b oh wter blks lnfgr whr bhr heads heads cosr sinr sposr snegr sclr maskr))
             (lout (nlga-keep b logits oner)))
        (nlga-seed-ce b logits ohtgt)
        (nlga-finish b (nlga-scalar b lr))
        (nlga-compile b)
        (list :cpu cpu :b b :oh oh :ohtgt ohtgt :lout lout :seq seq :vocab vocab)))))

(defun nl-llm-agent-ondevice-train (ctx trajs epochs)
  "Train the GPU graph in CTX on TRAJS (each a char-id list) for EPOCHS passes."
  (let ((b (plist-get ctx :b)) (oh (plist-get ctx :oh)) (ohtgt (plist-get ctx :ohtgt))
        (seq (plist-get ctx :seq)) (vocab (plist-get ctx :vocab)) (lout (plist-get ctx :lout)))
    (dotimes (_ epochs)
      (dolist (tr trajs)
        (nlga-update oh (nl-llm-agent--onehot-pad tr seq vocab))
        (nlga-update ohtgt (nl-llm-agent--onehot-pad (nl-llm-agent--shift-pad tr seq) seq vocab))
        (nth lout (nlga-step b))))))

(defun nl-llm-agent-ondevice-sync (ctx)
  "Transfer the GPU-trained weights back into the shared host tensors -- after this
the CPU rollout decodes with the GPU-trained weights."
  (nlga-readback (plist-get ctx :b)))

(defun nl-llm-agent-ondevice-free (ctx)
  (nlga-free (plist-get ctx :b)))

;;;###autoload
(cl-defun nl-llm-agent-improve-ondevice (ctx grammar reward &key (rounds 5) (rollouts 24) (epochs 4) (temp 1.0) (eval-n 20) trace)
  "Run the whole self-improvement loop on-device: each round, ROLLOUT actions on
the CPU view of CTX, keep those REWARD scores >= 1, TRAIN the shared nlga graph on
them on the GPU, then READBACK the trained weights into the CPU view.  The next
round's rollout therefore uses the GPU-trained weights.  Returns the per-round
success-rate list (initial + one per round)."
  (let ((m (plist-get ctx :cpu)))
    (cl-flet ((rate () (let ((ok 0)) (dotimes (_ eval-n)
                          (when (>= (funcall reward (car (nl-llm-agent-p5-rollout m grammar temp))) 1.0) (setq ok (1+ ok))))
                          (/ (float ok) eval-n))))
      ;; a replay buffer of recent wins keeps the GPU training stable -- training
      ;; only on the current round's wins overfits and the rollout oscillates.
      (let ((rates (list (rate))) (replay nil) (cap 48))
        (dotimes (r rounds)
          (dotimes (_ rollouts)
            (let ((roll (nl-llm-agent-p5-rollout m grammar temp)))
              (when (>= (funcall reward (car roll)) 1.0) (push (cdr roll) replay))))
          (when (> (length replay) cap) (setq replay (cl-subseq replay 0 cap)))
          (when replay
            (nl-llm-agent-ondevice-train ctx replay epochs)
            (nl-llm-agent-ondevice-sync ctx))
          (let ((rt (rate))) (push rt rates) (when trace (funcall trace (1+ r) (length replay) rt))))
        (nreverse rates)))))

(provide 'nl-llm-agent-ondevice)
;;; nl-llm-agent-ondevice.el ends here
