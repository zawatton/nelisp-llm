;;; bonsai-train.el --- a LoRA step on Ternary Bonsai 2 27B -*- lexical-binding: t -*-

;; Adapters go on the top NL_BONSAI_TRAIN_LAYERS blocks only, so the gradient
;; stops at that boundary and the 60 blocks below it are a constant for a given
;; prompt.  Their output is computed once and cached on disk: at fourteen
;; seconds a block that prefix is a quarter of an hour, and paying it per step
;; would make every other measurement here a measurement of it.
;;
;; A control prompt the training example says nothing about is scored every
;; few steps.  It is the only instrument that separates adaptation from damage:
;; a run that is destroying the model has a falling loss, a rising target
;; probability and a shrinking gradient, exactly like a run that is working.
;;
;; Usage: emacs -Q --batch -L lisp -L <photon> -l tools/bonsai-train.el

(require 'nl-llm-bonsai)
(require 'nl-llm-bonsai-backward)
(require 'nl-llm-weights-gpu)
(require 'nl-llm-lora)
(require 'nelisp-gpu-server)
(require 'cl-lib)

(defun bt--env (name default)
  (let ((v (getenv name))) (if (and v (> (length v) 0)) v default)))

(defun bt--ids (s) (mapcar #'string-to-number (split-string s "[ ,]+" t)))

(defun bt--vocab (path)
  (when (and path (file-readable-p path))
    (let ((v (make-vector 300000 nil)))
      (with-temp-buffer
        (let ((coding-system-for-read 'utf-8)) (insert-file-contents path))
        (goto-char (point-min))
        (while (not (eobp))
          (let* ((eol (line-end-position))
                 (tab (save-excursion (search-forward "\t" eol t))))
            (when tab
              (let ((id (string-to-number
                         (buffer-substring-no-properties (point) (1- tab)))))
                (when (< id (length v))
                  (aset v id (buffer-substring-no-properties tab eol))))))
          (forward-line 1)))
      v)))

(defun bt--show (vocab id)
  (let ((s (and vocab (< id (length vocab)) (aref vocab id))))
    (if s (format "%d %S" id (string-replace "Ġ" " " s)) (format "%d" id))))

(defun bt--save (path v)
  (let ((coding-system-for-write 'binary))
    (write-region (nelisp-gpu--floats-bytes (list v)) nil path nil 'silent)))

(defun bt--load (path n)
  (let ((raw (with-temp-buffer
               (set-buffer-multibyte nil)
               (let ((coding-system-for-read 'binary))
                 (insert-file-contents-literally path))
               (buffer-substring-no-properties (point-min) (point-max))))
        (out (make-vector n 0.0)))
    (dotimes (i n)
      (aset out i (nelisp-gpu--bits-f32
                   (logior (aref raw (* i 4))
                           (ash (aref raw (+ (* i 4) 1)) 8)
                           (ash (aref raw (+ (* i 4) 2)) 16)
                           (ash (aref raw (+ (* i 4) 3)) 24)))))
    out))

(defun bt--prefix (sess ids boundary cache tag)
  "Run blocks 0..BOUNDARY-1 over IDS and return the activation, caching it."
  (let* ((cfg (plist-get sess :cfg)) (dim (plist-get cfg :dim))
         (wts (plist-get sess :wts)) (seq (length ids))
         (path (and cache (expand-file-name
                           (format "prefix-%s-%d-%s.f32" tag boundary
                                   (if (plist-get sess :invert) "inv" "fwd"))
                           cache))))
    (if (and path (file-readable-p path))
        (progn (message "  prefix %-8s cached (%s)" tag
                        (file-name-nondirectory path))
               (bt--load path (* seq dim)))
      (let ((x (make-vector (* seq dim) 0.0)) (i 0) (t0 (float-time)))
        (dolist (tk ids)
          (let ((e (nl-llm-weights-embed wts tk)))
            (dotimes (j dim) (aset x (+ (* i dim) j) (aref e j))))
          (setq i (1+ i)))
        (dotimes (ly boundary)
          (let* ((lins (nl-llm-bonsai-linears sess ly))
                 (tbl (make-hash-table :test 'eq)))
            (cl-loop for (_k v) on lins by #'cddr
                     do (puthash v (nl-llm-wgpu-upload-lin v) tbl))
            (unwind-protect
                (let ((nl-llm-wgpu--transposes tbl)
                      (nl-llm-bonsai-apply-fn #'nl-llm-wgpu-apply-resident))
                  (setq x (nl-llm-bonsai-block sess ly x seq)))
              (nl-llm-wgpu-free-transposes tbl)
              (nl-llm-bonsai-forget-layer sess ly))
            (when (zerop (mod (1+ ly) 8))
              (message "  prefix %-8s block %2d/%d  (%.0fs)" tag (1+ ly)
                       boundary (- (float-time) t0)))))
        (when path (bt--save path x))
        (message "  prefix %-8s %d blocks in %.0fs" tag boundary
                 (- (float-time) t0))
        x))))

(defun bt--softmax-ce (logits target)
  "Return (LOSS DLOGITS RANK PROB) for cross-entropy at TARGET."
  (let* ((n (length logits)) (mx (aref logits 0)))
    (dotimes (i n) (when (> (aref logits i) mx) (setq mx (aref logits i))))
    (let ((sum 0.0) (p (make-vector n 0.0)) (rank 1))
      (dotimes (i n)
        (let ((e (exp (- (aref logits i) mx))))
          (aset p i e) (setq sum (+ sum e))))
      (dotimes (i n) (aset p i (/ (aref p i) sum)))
      (dotimes (i n) (when (> (aref logits i) (aref logits target))
                       (setq rank (1+ rank))))
      (let ((d (copy-sequence p)))
        (aset d target (- (aref d target) 1.0))
        (list (- (log (max 1.0e-30 (aref p target)))) d rank (aref p target))))))

(defun bt--top-forward (bc sess boundary nlayers x seq)
  "Forward the trainable blocks, returning (X TAPES)."
  (let ((tapes nil))
    (cl-loop for ly from boundary below nlayers
             do (let ((r (nl-llm-bonsai-bw-block-forward bc ly x seq)))
                  (setq x (nth 0 r))
                  (push (nth 1 r) tapes)))
    (list x (nreverse tapes))))

(defun bt-run ()
  (let* ((path (bt--env "NL_BONSAI_WTS" "build/bonsai/wts-full.bin"))
         (invert (and (getenv "NL_BONSAI_INVERT")
                      (> (length (getenv "NL_BONSAI_INVERT")) 0)))
         (sess (nl-llm-bonsai-open path invert))
         (cfg (plist-get sess :cfg))
         (dim (plist-get cfg :dim)) (nlayers (plist-get cfg :layers))
         (iv (plist-get cfg :full-attention-interval))
         (k (string-to-number (bt--env "NL_BONSAI_TRAIN_LAYERS" "4")))
         (boundary (max 0 (- nlayers k)))
         (steps (string-to-number (bt--env "NL_BONSAI_STEPS" "12")))
         (lr (string-to-number (bt--env "NL_BONSAI_LR" "0.002")))
         (rank (string-to-number (bt--env "NL_BONSAI_RANK" "8")))
         (cache (bt--env "NL_BONSAI_CACHE" "build/bonsai/cache"))
         (vocab (bt--vocab (getenv "NL_BONSAI_VOCAB")))
         (ids (bt--ids (bt--env "NL_BONSAI_PROMPT" "760 6511 314 9338 369")))
         (target (string-to-number (bt--env "NL_BONSAI_TARGET" "12095")))
         (cids (bt--ids (bt--env "NL_BONSAI_CONTROL" "760 6511 314 10884 369")))
         (ctarget (string-to-number (bt--env "NL_BONSAI_CONTROL_TARGET" "12095")))
         (seq (length ids)) (cseq (length cids)))
    (when cache (make-directory cache t))
    (message "%s  %d layers, training the top %d (blocks %d-%d), rotation %s"
             (file-name-nondirectory path) nlayers k boundary (1- nlayers)
             (if invert "inverse" "forward"))
    (message "  train   %s  ->  %s"
             (mapconcat (lambda (i) (bt--show vocab i)) ids " ")
             (bt--show vocab target))
    (message "  control %s  ->  %s"
             (mapconcat (lambda (i) (bt--show vocab i)) cids " ")
             (bt--show vocab ctarget))
    (nelisp-gpu-server-start)
    (unwind-protect
        (let ((x0 (bt--prefix sess ids boundary cache "train"))
              (xc (bt--prefix sess cids boundary cache "control"))
              (tbl (make-hash-table :test 'eq))
              (loras (make-hash-table :test 'equal)))
          ;; the trainable blocks and the head stay resident for every step
          (let ((t0 (float-time)))
            (cl-loop for ly from boundary below nlayers
                     do (cl-loop for (_k v) on (nl-llm-bonsai-linears sess ly)
                                 by #'cddr
                                 do (puthash v (nl-llm-wgpu-upload-lin v) tbl)))
            (puthash (nl-llm-bonsai-head sess)
                     (nl-llm-wgpu-upload-lin (nl-llm-bonsai-head sess)) tbl)
            (message "  resident %d buffers in %.1fs" (hash-table-count tbl)
                     (- (float-time) t0)))
          (unwind-protect
              (let* ((nl-llm-wgpu--transposes tbl)
                     (apply-fn #'nl-llm-wgpu-apply-resident)
                     (wt-fn #'nl-llm-wgpu-transpose)
                     (roles-dn '(:wqkv :wz :wout :wg :wu :wd))
                     (roles-at '(:wq :wk :wv :wo :wg :wu :wd)))
                (cl-loop for ly from boundary below nlayers
                         do (let ((lins (nl-llm-bonsai-linears sess ly)))
                              (dolist (role (if (= (mod ly iv) (1- iv))
                                                roles-at roles-dn))
                                (let ((lin (plist-get lins role)))
                                  (puthash (cons ly role)
                                           (nl-llm-lora-make
                                            (nl-llm-weights-lin-rows lin)
                                            (nl-llm-weights-lin-cols lin) rank)
                                           loras)))))
                (message "  %d adapters of rank %d on %d blocks"
                         (hash-table-count loras) rank k)
                ;; the control's standing, before anything is trained
                (let* ((bc (nl-llm-bonsai-bw-make sess loras apply-fn wt-fn))
                       (r (bt--top-forward bc sess boundary nlayers
                                           (copy-sequence xc) cseq))
                       (lg (nl-llm-bonsai-logits sess (nth 0 r) cseq))
                       (ce (bt--softmax-ce lg ctarget)))
                  (message "  control before  loss %.4f  rank %d  p %.5f  top %s"
                           (nth 0 ce) (nth 2 ce) (nth 3 ce)
                           (bt--show vocab (car (nl-llm-bonsai-argmax lg)))))
                (let ((first nil) (last nil))
                  (dotimes (step steps)
                    (let* ((t0 (float-time))
                           (bc (nl-llm-bonsai-bw-make sess loras apply-fn wt-fn))
                           (r (bt--top-forward bc sess boundary nlayers
                                               (copy-sequence x0) seq))
                           (xt (nth 0 r)) (tapes (nth 1 r))
                           (lg (nl-llm-bonsai-logits sess xt seq))
                           (ce (bt--softmax-ce lg target))
                           (dx (nl-llm-bonsai-bw-logits-backward
                                bc xt seq (nth 1 ce))))
                      (unless first (setq first (nth 0 ce)))
                      (setq last (nth 0 ce))
                      (cl-loop for tape in (reverse tapes)
                               do (setq dx (nl-llm-bonsai-bw-block-backward
                                            bc tape dx seq)))
                      (maphash
                       (lambda (key lora)
                         (let ((g (gethash key (plist-get bc :grads))))
                           (when g (nl-llm-wlora-sgd lora g lr))))
                       loras)
                      (message "  step %2d  loss %.4f  rank %d  p %.5f  top %-18s (%.0fs)"
                               step (nth 0 ce) (nth 2 ce) (nth 3 ce)
                               (bt--show vocab (car (nl-llm-bonsai-argmax lg)))
                               (- (float-time) t0))))
                  (let* ((bc (nl-llm-bonsai-bw-make sess loras apply-fn wt-fn))
                         (r (bt--top-forward bc sess boundary nlayers
                                             (copy-sequence xc) cseq))
                         (lg (nl-llm-bonsai-logits sess (nth 0 r) cseq))
                         (ce (bt--softmax-ce lg ctarget)))
                    (message "  control after   loss %.4f  rank %d  p %.5f  top %s"
                             (nth 0 ce) (nth 2 ce) (nth 3 ce)
                             (bt--show vocab (car (nl-llm-bonsai-argmax lg)))))
                  (message "  training loss %.4f -> %.4f over %d steps"
                           first last steps)))
            (nl-llm-wgpu-free-transposes tbl)))
      (nelisp-gpu-server-stop))
    (message "DONE")))

(bt-run)
