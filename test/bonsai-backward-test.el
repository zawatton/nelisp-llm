;;; bonsai-backward-test.el --- the hybrid block's gradients -*- lexical-binding: t -*-

;; Checked against finite differences on a model small enough to afford them
;; (tools/bonsai-synth.py), because every cheaper instrument here is blind:
;; the rotations are orthogonal, so a gradient that comes back through the
;; wrong one has exactly the right magnitude, and the gates read a different
;; basis from the projections beside them, so a mix-up changes an answer
;; without changing a shape.
;;
;; The taped forward is also checked for EXACT equality against the plain one.
;; Two implementations of the same block will drift otherwise, and a drift
;; that only shows up as a slightly wrong gradient is the kind that survives.

(require 'nl-llm-bonsai)
(require 'nl-llm-bonsai-backward)
(require 'nl-llm-lora)
(require 'nl-llm-weights-lora)
(require 'cl-lib)

(defvar bbt-pass 0)
(defvar bbt-fail 0)
(defvar bbt-model (or (getenv "NL_BONSAI_SYNTH") "build/bonsai-synth.bin"))

(defun bbt-check (name ok fmt &rest args)
  "Record and print a check.  FMT is the evidence when it passes and the
complaint when it does not, so a line never reads as its own opposite."
  (if ok (setq bbt-pass (1+ bbt-pass))
    (setq bbt-fail (1+ bbt-fail)))
  (message "%-52s %s  %s" name (if ok "PASS" "FAIL")
           (if (and (null args) (not ok)) fmt (apply #'format fmt args))))

(defun bbt--rand (n seed)
  (let ((v (make-vector n 0.0)) (s seed))
    (dotimes (i n)
      (setq s (mod (+ (* s 1103515245) 12345) 2147483648))
      (aset v i (- (/ (float s) 1073741824.0) 1.0)))
    v))

(defun bbt--dot (a b)
  (let ((s 0.0)) (dotimes (i (length a)) (setq s (+ s (* (aref a i) (aref b i))))) s))

(defun bbt--loss (bc layer x seq w)
  (bbt--dot (nth 0 (nl-llm-bonsai-bw-block-forward bc layer x seq)) w))

(defun bbt-run ()
  (unless (file-readable-p bbt-model)
    (message "bonsai-backward: no %s -- run `make bonsai-synth'" bbt-model)
    (kill-emacs 1))
  (let* ((sess (nl-llm-bonsai-open bbt-model))
         (cfg (plist-get sess :cfg))
         (dim (plist-get cfg :dim))
         (seq 4)
         (x (bbt--rand (* seq dim) 12345))
         (w (bbt--rand (* seq dim) 999)))

    ;; 1. the taped forward is the same function as the plain one
    (dotimes (ly (plist-get cfg :layers))
      (let* ((bc (nl-llm-bonsai-bw-make sess))
             (plain (nl-llm-bonsai-block sess ly (copy-sequence x) seq))
             (taped (nth 0 (nl-llm-bonsai-bw-block-forward
                            bc ly (copy-sequence x) seq)))
             (worst 0.0))
        (dotimes (i (length plain))
          (setq worst (max worst (abs (- (aref plain i) (aref taped i))))))
        (bbt-check (format "layer %d: taped forward == plain forward" ly)
                   (= worst 0.0) "worst |difference| %.3e" worst)))

    ;; 2. dX against central differences
    (dotimes (ly (plist-get cfg :layers))
      (let* ((bc (nl-llm-bonsai-bw-make sess))
             (fw (nl-llm-bonsai-bw-block-forward bc ly (copy-sequence x) seq))
             (dx (nl-llm-bonsai-bw-block-backward bc (nth 1 fw) w seq))
             (h 1.0e-4) (worst 0.0) (worst-at -1) (probed 0))
        ;; a spread of coordinates rather than all of them: positions matter
        ;; here, because the recurrence and the causal mask make the gradient
        ;; at the last position unlike the gradient at the first
        (dolist (j (list 0 3 (1- dim) dim (+ dim 7)
                         (* 2 dim) (+ (* 2 dim) 5)
                         (* 3 dim) (+ (* 3 dim) dim -1)))
          (let* ((xp (copy-sequence x)) (xm (copy-sequence x)))
            (aset xp j (+ (aref x j) h))
            (aset xm j (- (aref x j) h))
            (let* ((lp (bbt--loss (nl-llm-bonsai-bw-make sess) ly xp seq w))
                   (lm (bbt--loss (nl-llm-bonsai-bw-make sess) ly xm seq w))
                   (num (/ (- lp lm) (* 2.0 h)))
                   (ana (aref dx j))
                   (rel (/ (abs (- num ana))
                           (max 1.0e-8 (abs num) (abs ana)))))
              (setq probed (1+ probed))
              (when (> rel worst) (setq worst rel worst-at j)))))
        (bbt-check (format "layer %d: dX matches finite differences" ly)
                   (< worst 2.0e-4) "worst rel %.3e at %d over %d probes"
                   worst worst-at probed)))

    ;; 3. a control: the gradient must not survive the block being changed.
    ;; Without this, a dX of the right shape but the wrong content passes 2
    ;; whenever the numbers happen to be small.
    (let* ((bc (nl-llm-bonsai-bw-make sess))
           (fw (nl-llm-bonsai-bw-block-forward bc 0 (copy-sequence x) seq))
           (dx (nl-llm-bonsai-bw-block-backward bc (nth 1 fw) w seq))
           (bc2 (nl-llm-bonsai-bw-make sess))
           (fw2 (nl-llm-bonsai-bw-block-forward bc2 1 (copy-sequence x) seq))
           (dx2 (nl-llm-bonsai-bw-block-backward bc2 (nth 1 fw2) w seq))
           (same t))
      (dotimes (i (length dx))
        (when (> (abs (- (aref dx i) (aref dx2 i))) 1.0e-12) (setq same nil)))
      (bbt-check "control: the two block types differ" (not same)
                 (if same "a DeltaNet block and an attention block gave the same dX"
                   "the two block types disagree, as they must")))

    ;; 4. adapter gradients against finite differences
    (dotimes (ly (plist-get cfg :layers))
      (let* ((iv (plist-get cfg :full-attention-interval))
             (role (if (= (mod ly iv) (1- iv)) :wo :wout))
             (lins (nl-llm-bonsai-linears sess ly))
             (lin (plist-get lins role))
             (lora (nl-llm-lora-make (nl-llm-weights-lin-rows lin)
                                     (nl-llm-weights-lin-cols lin) 2))
             (bdata (photon-tensor-data (plist-get lora :b))))
        ;; a fresh adapter has B = 0, which makes every dB numerically zero;
        ;; perturb it so the check has something to measure
        (dotimes (i (length bdata))
          (aset bdata i (* 0.05 (aref (bbt--rand (length bdata) 4242) i))))
        (let* ((loras (nl-llm-bonsai-bw-loras (list ly role lora)))
               (bc (nl-llm-bonsai-bw-make sess loras))
               (fw (nl-llm-bonsai-bw-block-forward bc ly (copy-sequence x) seq))
               (_ (nl-llm-bonsai-bw-block-backward bc (nth 1 fw) w seq))
               (g (nl-llm-bonsai-bw-grad bc ly role))
               (h 1.0e-4) (worst 0.0) (kind ""))
          (dolist (slot '(:a :b))
            (let* ((data (photon-tensor-data (plist-get lora slot)))
                   (grad (plist-get g (if (eq slot :a) :da :db))))
              (dolist (j (list 0 1 (/ (length data) 2) (1- (length data))))
                (let ((save (aref data j)))
                  (aset data j (+ save h))
                  (let ((lp (bbt--loss (nl-llm-bonsai-bw-make sess loras)
                                       ly x seq w)))
                    (aset data j (- save h))
                    (let* ((lm (bbt--loss (nl-llm-bonsai-bw-make sess loras)
                                          ly x seq w))
                           (num (/ (- lp lm) (* 2.0 h)))
                           (ana (aref grad j))
                           (rel (/ (abs (- num ana))
                                   (max 1.0e-8 (abs num) (abs ana)))))
                      (when (> rel worst) (setq worst rel kind (format "%s[%d]" slot j)))))
                  (aset data j save)))))
          (bbt-check (format "layer %d: adapter %s gradients" ly role)
                     (< worst 2.0e-4) "worst rel %.3e at %s" worst kind))))

    ;; 4b. the same gradients with the norm gains applied rather than folded.
    ;; With `nl-llm-bonsai-folded-gains' on -- the default, because the real
    ;; file has them folded -- nothing in this suite ever multiplies by a gain,
    ;; so the branch that does would stop working unnoticed.
    (let ((nl-llm-bonsai-folded-gains nil))
      (dotimes (ly (plist-get cfg :layers))
        (let* ((bc (nl-llm-bonsai-bw-make sess))
               (fw (nl-llm-bonsai-bw-block-forward bc ly (copy-sequence x) seq))
               (dx (nl-llm-bonsai-bw-block-backward bc (nth 1 fw) w seq))
               (h 1.0e-4) (worst 0.0))
          (dolist (j (list 0 dim (+ (* 2 dim) 5) (+ (* 3 dim) dim -1)))
            (let ((xp (copy-sequence x)) (xm (copy-sequence x)))
              (aset xp j (+ (aref x j) h))
              (aset xm j (- (aref x j) h))
              (let* ((lp (bbt--loss (nl-llm-bonsai-bw-make sess) ly xp seq w))
                     (lm (bbt--loss (nl-llm-bonsai-bw-make sess) ly xm seq w))
                     (num (/ (- lp lm) (* 2.0 h)))
                     (rel (/ (abs (- num (aref dx j)))
                             (max 1.0e-8 (abs num) (abs (aref dx j))))))
                (setq worst (max worst rel)))))
          (bbt-check (format "layer %d: dX with the gains applied" ly)
                     (< worst 2.0e-4) "worst rel %.3e" worst))))
    ;; and a control: folding must actually change the answer
    (let* ((a (nl-llm-bonsai-block sess 0 (copy-sequence x) seq))
           (b (let ((nl-llm-bonsai-folded-gains nil))
                (nl-llm-bonsai-block sess 0 (copy-sequence x) seq)))
           (worst 0.0))
      (dotimes (i (length a))
        (setq worst (max worst (abs (- (aref a i) (aref b i))))))
      (bbt-check "control: folding the gains changes the block" (> worst 1.0e-6)
                 "worst |difference| %.3e" worst))

    ;; 5. the head and the final norm
    (let* ((bc (nl-llm-bonsai-bw-make sess))
           (vocab (plist-get cfg :vocab))
           (dl (bbt--rand vocab 31337))
           (lossfn (lambda (xx)
                     (bbt--dot (nl-llm-bonsai-logits sess xx seq) dl)))
           (dx (nl-llm-bonsai-bw-logits-backward bc x seq dl))
           (h 1.0e-4) (worst 0.0))
      (dolist (j (list (* 3 dim) (+ (* 3 dim) 1) (+ (* 3 dim) dim -1)))
        (let ((xp (copy-sequence x)) (xm (copy-sequence x)))
          (aset xp j (+ (aref x j) h))
          (aset xm j (- (aref x j) h))
          (let* ((num (/ (- (funcall lossfn xp) (funcall lossfn xm)) (* 2.0 h)))
                 (rel (/ (abs (- num (aref dx j)))
                         (max 1.0e-8 (abs num) (abs (aref dx j))))))
            (setq worst (max worst rel)))))
      (bbt-check "head + final norm: dX matches finite differences"
                 (< worst 2.0e-4) "worst rel %.3e" worst)
      ;; the positions the head did not read must get exactly nothing
      (let ((leak 0.0))
        (dotimes (i (* 3 dim)) (setq leak (max leak (abs (aref dx i)))))
        (bbt-check "head: earlier positions get no gradient" (= leak 0.0)
                   "worst |dX| before the read position %.3e" leak)))

    ;; --- the same gradients on the donor's real weights ---------------
    ;;
    ;; The synthetic model has the right shapes and arbitrary numbers in them.
    ;; The donor has numbers a real training run produced, is checked layer by
    ;; layer against tools/qwen-forward-ref.py, and exercises the branches the
    ;; synthetic one cannot: gains applied rather than folded, no rotation, an
    ;; ungated attn_q, a tied head, and a rotary over the whole head.
    (let ((donor "build/donor/qwen3-0.6b/weights.bin"))
      (if (not (file-readable-p donor))
          (message "%-52s ----  no donor at %s" "donor gradients" donor)
        (let* ((ds (nl-llm-bonsai-open donor))
               (dcfg (plist-get ds :cfg))
               (ddim (plist-get dcfg :dim))
               (dseq 4)
               (dx (bbt--rand (* dseq ddim) 4242))
               (dw (bbt--rand (* dseq ddim) 31)))
          (bbt-check "donor: no rotation, gains applied, ungated q"
                     (and (null (plist-get ds :rotate))
                          (null (plist-get ds :folded))
                          (plist-get dcfg :tied-head))
                     "rotate %S folded %S tied %S"
                     (plist-get ds :rotate) (plist-get ds :folded)
                     (plist-get dcfg :tied-head))
          (dotimes (ly 2)
            (let* ((bc (nl-llm-bonsai-bw-make ds))
                   (plain (nl-llm-bonsai-block ds ly (copy-sequence dx) dseq))
                   (taped (nth 0 (nl-llm-bonsai-bw-block-forward
                                  bc ly (copy-sequence dx) dseq)))
                   (worst 0.0))
              (dotimes (i (length plain))
                (setq worst (max worst (abs (- (aref plain i) (aref taped i))))))
              (bbt-check (format "donor layer %d: taped == plain" ly)
                         (= worst 0.0) "worst |difference| %.3e" worst)))
          (dotimes (ly 2)
            (let* ((bc (nl-llm-bonsai-bw-make ds))
                   (fw (nl-llm-bonsai-bw-block-forward bc ly (copy-sequence dx) dseq))
                   (grad (nl-llm-bonsai-bw-block-backward bc (nth 1 fw) dw dseq))
                   (h 1.0e-4) (worst 0.0) (at -1))
              (dolist (j (list 0 7 (1- ddim) ddim (+ (* 2 ddim) 13)
                               (+ (* 3 ddim) ddim -1)))
                (let ((xp (copy-sequence dx)) (xm (copy-sequence dx)))
                  (aset xp j (+ (aref dx j) h))
                  (aset xm j (- (aref dx j) h))
                  (let* ((lp (bbt--dot (nth 0 (nl-llm-bonsai-bw-block-forward
                                               (nl-llm-bonsai-bw-make ds) ly xp dseq))
                                       dw))
                         (lm (bbt--dot (nth 0 (nl-llm-bonsai-bw-block-forward
                                               (nl-llm-bonsai-bw-make ds) ly xm dseq))
                                       dw))
                         (num (/ (- lp lm) (* 2.0 h)))
                         (rel (/ (abs (- num (aref grad j)))
                                 (max 1.0e-8 (abs num) (abs (aref grad j))))))
                    (when (> rel worst) (setq worst rel at j)))))
              (bbt-check (format "donor layer %d: dX matches finite differences" ly)
                         (< worst 5.0e-4) "worst rel %.3e at %d" worst at))))))

    (message "bonsai-backward: %d passed, %d failed" bbt-pass bbt-fail)
    (when (> bbt-fail 0) (kill-emacs 1))))

(bbt-run)
