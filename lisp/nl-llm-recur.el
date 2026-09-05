;;; nl-llm-recur.el --- recurrent depth: latent reasoning by iterating a shared block  -*- lexical-binding: t; -*-

;; Implements Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson,
;; Kailkhura, Bhatele, Goldstein, "Scaling up Test-Time Compute with Latent
;; Reasoning: A Recurrent Depth Approach" (arXiv 2502.05171, Feb 2025) on the
;; nl-llm CPU autograd path (docs/design/07-recurrent-depth.org).
;;
;; A model is three stacks of plain pre-norm GQA/SwiGLU blocks
;; (`nl-llm-ag-block'): a PRELUDE that embeds the tokens into E (seq x dim), a
;; recurrent CORE that iterates a latent state S -- injecting E at every
;; iteration through a linear ADAPTER on the column-concat [S ; E] -- and a
;; CODA that reads the final state into logits.  Training draws R from a
;; log-normal Poisson (`nl-llm-recur-sample-r') and backpropagates only
;; through the LAST K iterations (truncated BPTT): the first R-K iterations
;; run inside `nl-llm-ag-no-grad' and are re-attached to the tape as a fresh
;; leaf via `photon-autograd-const', so gradient never reaches further back
;; than that -- except through E, whose own tape entries were recorded before
;; the no-grad scope and so still carry gradient into the prelude via the
;; injection at each of the surviving K iterations.  Inference can run at any
;; fixed R, or adaptively (`nl-llm-recur-forward-adaptive'): stop iterating as
;; soon as the coda's output distribution stops moving.
;;
;; `nl-llm-recur-task-sum-mod' is the synthetic task used to show depth
;; actually buys something: K random digits plus a separator, target = the
;; digit sum mod 10, with no intermediate reasoning steps in the input -- the
;; model must do the K-1 additions somewhere inside the recurrence.

;;; Code:

(require 'cl-lib)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-autograd)

;; --- truncated-BPTT no-grad tape scope --------------------------------

;;;###autoload
(defmacro nl-llm-ag-no-grad (&rest body)
  "Run BODY, then drop every tape entry it recorded (no gradient flows into
or out of the ops recorded inside).  The value is still a pav; detach it
with `photon-autograd-const' on its value before using it downstream."
  `(let ((nl-llm--tape0 photon-autograd--tape))
     (prog1 (progn ,@body) (setq photon-autograd--tape nl-llm--tape0))))

;; --- deterministic hashing (no dependence on Emacs' mutable RNG state) -

(defun nl-llm-recur--hash01 (i seed)
  "Deterministic pseudo-uniform float in the open interval (0,1) derived
from integer I and SEED via a multiplicative integer hash (portable,
side-effect-free -- unlike Emacs' global `random' state)."
  (let ((h (mod (+ (* (+ i 1) 2654435761) (* (+ seed 1) 40503)
                   (* (+ i 1) (+ i 1) 97))
               1000003)))
    (/ (+ 1.0 (float h)) 1000005.0)))

(defun nl-llm-recur--hash-digit (i seed)
  "Deterministic pseudo-random digit in [0,9] from integer I and SEED."
  (mod (+ (* (+ i 1) 2654435761) (* (+ seed 1) 40503) (* i 97)) 10))

;;;###autoload
(defun nl-llm-recur-randn (n sigma seed)
  "Return a fresh Lisp vector of N floats i.i.d. ~ N(0, SIGMA^2), deterministic
from SEED.  Uses the Marsaglia polar method (only `sqrt' / `log', no `cos' /
`sin') over `nl-llm-recur--hash01' uniforms, so it needs no mutable RNG state
and stays reproducible on the NeLisp standalone reader."
  (let ((out (make-vector n 0.0)) (i 0) (k 0))
    (while (< i n)
      (let ((found nil) (u 0.0) (v 0.0) (s2 0.0))
        (while (not found)
          (setq u (- (* 2.0 (nl-llm-recur--hash01 (* 2 k) seed)) 1.0))
          (setq v (- (* 2.0 (nl-llm-recur--hash01 (1+ (* 2 k)) seed)) 1.0))
          (setq s2 (+ (* u u) (* v v)))
          (setq k (1+ k))
          (when (and (> s2 1.0e-12) (< s2 1.0)) (setq found t)))
        (let ((mul (sqrt (/ (* -2.0 (log s2)) s2))))
          (aset out i (* sigma u mul))
          (setq i (1+ i))
          (when (< i n)
            (aset out i (* sigma v mul))
            (setq i (1+ i))))))
    out))

;; --- model: prelude / adapter / core / coda, all plain GQA+SwiGLU blocks

(defun nl-llm-recur--p (shape seed scale)
  "A pav leaf of SHAPE, deterministically hash-initialized from SEED, scaled
by SCALE (same hash formula used across this repo's examples/tests)."
  (let ((n 1)) (dolist (d shape) (setq n (* n d)))
    (photon-autograd-const
     (photon-tensor
      shape
      (let ((v (make-vector n 0.0)) (i 0))
        (while (< i n)
          (aset v i (* scale 2.0
                       (- (/ (float (mod (+ (* (1+ i) 2654435761) (* (1+ seed) 40503)) 65536))
                             65536.0)
                          0.5)))
          (setq i (1+ i)))
        v)))))

(defun nl-llm-recur--c (n val) (photon-autograd-const (photon-tensor (list n) (make-vector n val))))

(defun nl-llm-recur--block (dim ff heads kv-heads seed)
  "One plain pre-norm GQA/SwiGLU block plist (what `nl-llm-ag-block' consumes).
HEADS/KV-HEADS size the K/V projections: `nl-llm-ag-gqa' expects K/V to be
KV-HEADS*hd wide (hd = DIM/HEADS), not DIM wide -- see how
`examples/train-modern-full.el' builds its Wk/Wv with kvdim."
  (let* ((sc (/ 1.0 (sqrt (float dim))))
         (hd (/ dim heads)) (kvdim (* kv-heads hd)))
    (list :ln1g (nl-llm-recur--c dim 1.0)
          :wq (nl-llm-recur--p (list dim dim) (+ seed 1) sc) :bq (nl-llm-recur--c dim 0.0)
          :wk (nl-llm-recur--p (list kvdim dim) (+ seed 2) sc) :bk (nl-llm-recur--c kvdim 0.0)
          :wv (nl-llm-recur--p (list kvdim dim) (+ seed 3) sc) :bv (nl-llm-recur--c kvdim 0.0)
          :wo (nl-llm-recur--p (list dim dim) (+ seed 4) sc) :bo (nl-llm-recur--c dim 0.0)
          :ln2g (nl-llm-recur--c dim 1.0)
          :wg (nl-llm-recur--p (list ff dim) (+ seed 5) sc) :bg (nl-llm-recur--c ff 0.0)
          :wu (nl-llm-recur--p (list ff dim) (+ seed 6) sc) :bu (nl-llm-recur--c ff 0.0)
          :wd (nl-llm-recur--p (list dim ff) (+ seed 7) sc) :bd (nl-llm-recur--c dim 0.0))))

(defun nl-llm-recur--blocks (dim ff heads kv-heads n seed0)
  "A list of N blocks (see `nl-llm-recur--block'), each with a distinct seed."
  (let ((out nil) (i 0))
    (while (< i n)
      (push (nl-llm-recur--block dim ff heads kv-heads (+ seed0 (* i 20))) out)
      (setq i (1+ i)))
    (nreverse out)))

(defconst nl-llm-recur--block-keys
  '(:ln1g :wq :bq :wk :bk :wv :bv :wo :bo :ln2g :wg :bg :wu :bu :wd :bd)
  "The pav-valued keys of one block plist, in a fixed order.")

;;;###autoload
(cl-defun nl-llm-recur-model-new (&key (vocab 11) (dim 16) (heads 2) (kv-heads heads)
                                        (ff (* 2 dim)) (n-prelude 1) (n-core 1) (n-coda 1)
                                        (seed 1) (sigma 1.0))
  "Build a recurrent-depth model: a PRELUDE of N-PRELUDE blocks embeds the
tokens into E; a recurrent CORE of N-CORE blocks is iterated over a latent
state S, injecting E at every iteration through a linear ADAPTER (:wa :ba,
DIM x 2*DIM with bias) applied to the column-concat [S ; E]; a CODA of
N-CODA blocks plus a final RMSNorm and LM head reads the last state into
logits.  HEADS/KV-HEADS/FF are shared by every block.  SIGMA is the stddev
used for the default (fresh, per-forward) Gaussian init of S0.  Returns a
plist of pav params + dims:
:wte :prelude :wa :ba :core :coda :lnfg :wh :bh :dim :heads :kv-heads :sigma."
  (let ((sc (/ 1.0 (sqrt (float dim)))))
    (list :wte (nl-llm-recur--p (list vocab dim) seed sc)
          :prelude (nl-llm-recur--blocks dim ff heads kv-heads n-prelude (+ (* seed 1000) 100))
          :wa (nl-llm-recur--p (list dim (* 2 dim)) (+ (* seed 1000) 500) sc)
          :ba (nl-llm-recur--c dim 0.0)
          :core (nl-llm-recur--blocks dim ff heads kv-heads n-core (+ (* seed 1000) 600))
          :coda (nl-llm-recur--blocks dim ff heads kv-heads n-coda (+ (* seed 1000) 900))
          :lnfg (nl-llm-recur--c dim 1.0)
          :wh (nl-llm-recur--p (list vocab dim) (+ (* seed 1000) 999) sc)
          :bh (nl-llm-recur--c vocab 0.0)
          :dim dim :heads heads :kv-heads kv-heads :sigma sigma)))

;;;###autoload
(defun nl-llm-recur-params (model)
  "Flat list of every pav parameter in MODEL, for zero-grad / SGD."
  (append (list (plist-get model :wte))
          (nl-llm-recur--blocks-params (plist-get model :prelude))
          (list (plist-get model :wa) (plist-get model :ba))
          (nl-llm-recur--blocks-params (plist-get model :core))
          (nl-llm-recur--blocks-params (plist-get model :coda))
          (list (plist-get model :lnfg) (plist-get model :wh) (plist-get model :bh))))

(defun nl-llm-recur--blocks-params (blocks)
  "Flat list of every pav in BLOCKS (a list of block plists)."
  (let (out)
    (dolist (blk blocks)
      (dolist (k nl-llm-recur--block-keys) (push (plist-get blk k) out)))
    (nreverse out)))

;; --- one recurrent-core iteration --------------------------------------

(defun nl-llm-recur--step (s e model heads kv-heads)
  "One recurrent-core iteration: S_i = R(A([S_{i-1} ; E])), A the adapter
linear (`:wa' `:ba'), R the stack of MODEL's `:core' blocks."
  (let* ((cat (nl-llm-ag-concat-cols (list s e)))
         (h (photon-autograd-linear cat (plist-get model :wa) (plist-get model :ba))))
    (dolist (blk (plist-get model :core)) (setq h (nl-llm-ag-block h blk heads kv-heads)))
    h))

(defun nl-llm-recur--fresh-s0 (seq dim sigma)
  "A fresh (per-call) seeded Gaussian S0 pav (seq x dim): the SEED fed to the
deterministic `nl-llm-recur-randn' comes from Emacs' global `random' (real
entropy), so repeated calls differ, while the sampling arithmetic itself
stays the portable hash-based Marsaglia code."
  (photon-autograd-const
   (photon-tensor (list seq dim) (nl-llm-recur-randn (* seq dim) sigma (random 1000000000)))))

;; --- forward / loss -----------------------------------------------------

;;;###autoload
(cl-defun nl-llm-recur-forward (model tokens r &key k s0)
  "Recurrent-depth forward over TOKENS (a list of ids), iterating MODEL's core
R times.  Only the last K iterations (default R, i.e. full backprop) stay on
the autograd tape; the first R-K run inside `nl-llm-ag-no-grad' and the state
at that boundary is re-attached to the tape as a fresh leaf via
`photon-autograd-const', so no gradient flows further back through the
recurrence than that -- but E (the prelude's output, recorded before the
no-grad scope) keeps its own tape entries, so it still receives gradient
through the injection at each of the surviving K iterations.  S0 defaults to
a fresh seeded Gaussian draw (see `nl-llm-recur--fresh-s0'); pass an explicit
S0 pav where bit-identical reproducibility matters.  Returns a plist
\(:logits PAV :states (S_1-VALUE ... S_R-VALUE) :e PAV :r R)."
  (let* ((dim (plist-get model :dim)) (heads (plist-get model :heads))
         (kv-heads (plist-get model :kv-heads)) (sigma (plist-get model :sigma))
         (seq (length tokens)) (kk (max 0 (min r (or k r)))) (nograd (- r kk))
         (e (photon-autograd-embedding (plist-get model :wte) tokens dim))
         (s (or s0 (nl-llm-recur--fresh-s0 seq dim sigma)))
         (states nil))
    (dolist (blk (plist-get model :prelude)) (setq e (nl-llm-ag-block e blk heads kv-heads)))
    (when (> nograd 0)
      (nl-llm-ag-no-grad
       (let ((i 0))
         (while (< i nograd)
           (setq s (nl-llm-recur--step s e model heads kv-heads))
           (push (pav-value s) states)
           (setq i (1+ i)))))
      (setq s (photon-autograd-const (pav-value s))))
    (let ((i 0))
      (while (< i kk)
        (setq s (nl-llm-recur--step s e model heads kv-heads))
        (push (pav-value s) states)
        (setq i (1+ i))))
    (let ((coda-out s))
      (dolist (blk (plist-get model :coda)) (setq coda-out (nl-llm-ag-block coda-out blk heads kv-heads)))
      (let* ((normed (nl-llm-ag-rmsnorm coda-out (plist-get model :lnfg)))
             (logits (photon-autograd-linear normed (plist-get model :wh) (plist-get model :bh))))
        (list :logits logits :states (nreverse states) :e e :r r)))))

;;;###autoload
(cl-defun nl-llm-recur-loss (model tokens targets r &key k s0)
  "Softmax cross-entropy loss of `nl-llm-recur-forward' (MODEL TOKENS R
:K K :S0 S0) against per-position TARGETS (a vector, next-token convention)."
  (photon-autograd-softmax-ce
   (plist-get (nl-llm-recur-forward model tokens r :k k :s0 s0) :logits) targets))

;; --- training-time R sampler: log-normal Poisson, Knuth's method -------

(defun nl-llm-recur--u01 ()
  "Uniform float in the open interval (0,1) via Emacs' global `random'."
  (/ (+ 1.0 (float (random 1000000))) 1000002.0))

(defun nl-llm-recur--randn1 ()
  "One standard-normal draw (Marsaglia polar, `sqrt'/`log' only, no trig)
using Emacs' global `random' for uniforms -- seed it once via
`(random \"some-fixed-string\")' for a reproducible run."
  (let ((found nil) (u 0.0) (s2 0.0))
    (while (not found)
      (setq u (- (* 2.0 (nl-llm-recur--u01)) 1.0))
      (let ((v (- (* 2.0 (nl-llm-recur--u01)) 1.0)))
        (setq s2 (+ (* u u) (* v v)))
        (when (and (> s2 1.0e-12) (< s2 1.0)) (setq found t))))
    (* u (sqrt (/ (* -2.0 (log s2)) s2)))))

(defun nl-llm-recur--poisson-knuth (lam)
  "One Poisson(LAM) draw via Knuth's algorithm: only `exp' and uniform draws
\(no trig, no log).  LAM must be >= 0."
  (let ((l (exp (- (max 0.0 lam)))) (k 0) (p 1.0) (done nil))
    (while (not done)
      (setq k (1+ k))
      (setq p (* p (nl-llm-recur--u01)))
      (when (<= p l) (setq done t)))
    (1- k)))

;;;###autoload
(defun nl-llm-recur-sample-r (rbar &optional sigma-log)
  "Sample a training-time iteration count R = 1 + Poisson(TAU), TAU drawn
log-normal with mean (RBAR - 1) and log-sigma SIGMA-LOG (default 0.5).
Always >= 1.  The log-normal draw uses the Marsaglia polar method (no trig);
the Poisson draw uses Knuth's method (`exp' + uniforms, no trig).  Uses
Emacs' global `random'; seed it once via `(random \"...\")' for a
reproducible run."
  (let* ((sl (or sigma-log 0.5))
         (mean-tau (max 1.0e-3 (- (float rbar) 1.0)))
         (mu (- (log mean-tau) (* 0.5 sl sl)))
         (z (nl-llm-recur--randn1))
         ;; clamp: keeps `(exp (- tau))' well away from the NeLisp exp(-huge)
         ;; NaN edge instead of relying on an astronomically unlikely draw.
         (tau (min 700.0 (exp (+ mu (* sl z))))))
    (+ 1 (nl-llm-recur--poisson-knuth tau))))

;; --- adaptive-exit inference --------------------------------------------

(defun nl-llm-recur--coda-last-row-probs (s model heads kv-heads)
  "Run MODEL's `:coda' blocks + final RMSNorm + head on state S.  Returns
\(LOGITS-PAV . LAST-ROW-PROBS), LAST-ROW-PROBS the softmax distribution
\(a plain float vector) at S's last sequence position."
  (let ((h s))
    (dolist (blk (plist-get model :coda)) (setq h (nl-llm-ag-block h blk heads kv-heads)))
    (let* ((normed (nl-llm-ag-rmsnorm h (plist-get model :lnfg)))
           (logits (photon-autograd-linear normed (plist-get model :wh) (plist-get model :bh)))
           (lv (pav-value logits)) (sh (photon-tensor-shape lv))
           (seq (car sh)) (vocab (nth 1 sh)) (base (* (1- seq) vocab))
           (row (make-vector vocab 0.0)) (j 0))
      (while (< j vocab) (aset row j (aref (photon-tensor-data lv) (+ base j))) (setq j (1+ j)))
      (cons logits (photon-tensor-data (photon-tensor-softmax-rows (photon-tensor (list 1 vocab) row)))))))

(defun nl-llm-recur--kl (p q)
  "KL(P || Q) for two equal-length probability vectors (plain float vectors)."
  (let ((n (length p)) (i 0) (acc 0.0))
    (while (< i n)
      (let ((pv (aref p i)))
        (when (> pv 1.0e-12)
          (setq acc (+ acc (* pv (log (/ pv (max 1.0e-12 (aref q i)))))))))
      (setq i (1+ i)))
    acc))

;;;###autoload
(cl-defun nl-llm-recur-forward-adaptive (model tokens rmax eps &key s0)
  "Zero-shot adaptive-exit inference over TOKENS (no gradient): iterate
MODEL's recurrent core, and after every iteration -- plus once before any
iteration, on S0 itself, as the i=0 reference -- run the coda and take the
softmax distribution at the last sequence position.  Stop as soon as
KL(p_{i-1} || p_i) < EPS, or at RMAX.  Returns a plist (:logits PAV :r-used N
:kls (kl_1 ... kl_N)), LOGITS the coda logits of the state at exit."
  (let* ((dim (plist-get model :dim)) (heads (plist-get model :heads))
         (kv-heads (plist-get model :kv-heads)) (sigma (plist-get model :sigma))
         (seq (length tokens)) (logits-out nil) (r-used rmax) (kls nil) (stopped nil))
    (nl-llm-ag-no-grad
     (let* ((e (photon-autograd-embedding (plist-get model :wte) tokens dim))
            (s (or s0 (nl-llm-recur--fresh-s0 seq dim sigma))))
       (dolist (blk (plist-get model :prelude)) (setq e (nl-llm-ag-block e blk heads kv-heads)))
       (let* ((ref (nl-llm-recur--coda-last-row-probs s model heads kv-heads))
              (prev-p (cdr ref)) (i 1))
         (while (and (<= i rmax) (not stopped))
           (setq s (nl-llm-recur--step s e model heads kv-heads))
           (let* ((cur (nl-llm-recur--coda-last-row-probs s model heads kv-heads))
                  (kl (nl-llm-recur--kl prev-p (cdr cur))))
             (setq logits-out (car cur))
             (push kl kls)
             (setq prev-p (cdr cur))
             (setq r-used i)
             (when (< kl eps) (setq stopped t)))
           (setq i (1+ i))))))
    (list :logits logits-out :r-used r-used :kls (nreverse kls))))

;; --- synthetic task: K-digit sum mod 10, no intermediate steps ---------

;;;###autoload
(defun nl-llm-recur-task-sum-mod (k n seed)
  "Generate N examples of the depth-demanding sum-mod-10 task: K random digit
tokens (ids 0-9) followed by one separator token (id 10; vocab = 11).  The
target at every position is the usual next token, except the last (there is
no real token after the separator): there the target is the digit sum mod
10, so the model must have computed the K-1 additions somewhere inside the
recurrence by the time it reaches that position.  Deterministic from SEED.
Returns a list of N (TOKENS . TARGETS) conses: TOKENS a list of K+1 ids,
TARGETS a vector of K+1 ids."
  (let ((sep 10) (out nil) (ex 0))
    (while (< ex n)
      (let ((digits nil) (s 0) (j 0))
        (while (< j k)
          (let ((d (nl-llm-recur--hash-digit (+ (* ex 97) j) seed)))
            (push d digits) (setq s (+ s d)))
          (setq j (1+ j)))
        (setq digits (nreverse digits))
        (let* ((tokens (append digits (list sep)))
               (targets (make-vector (1+ k) 0)) (i 0))
          (while (< i (1- k))
            (aset targets i (nth (1+ i) digits))
            (setq i (1+ i)))
          (aset targets (1- k) sep)
          (aset targets k (mod s 10))
          (push (cons tokens targets) out)))
      (setq ex (1+ ex)))
    (nreverse out)))

(provide 'nl-llm-recur)
;;; nl-llm-recur.el ends here
