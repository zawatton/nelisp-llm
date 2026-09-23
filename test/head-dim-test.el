;;; head-dim-test.el --- attention with a head dim decoupled from dim/heads  -*- lexical-binding: t; -*-
;; Run: emacs -Q --batch -l test/head-dim-test.el
;;
;; Doc 08 Phase 2a.  Qwen3 sets head_dim independently of hidden_size /
;; num_attention_heads: Qwen3-0.6B is hidden 1024, 16 query heads, head_dim
;; 128, so q_proj is 1024 -> 2048 and o_proj is 2048 -> 1024.  Attention here
;; derived the head width as (/ dim heads) at every call site, which would give
;; 64, and the mismatch does not raise -- the projections are the right shape,
;; so the loops simply stride the wrong distance and produce plausible numbers.
;; That is why this suite carries its OWN reference implementation of decoupled
;; attention rather than comparing two nelisp-llm code paths against each other:
;; both paths shared the assumption, so they agreed while both were wrong.
;;
;; The layer/model plist key is =:head-dim=, absent meaning (/ dim heads), so
;; every existing model and test is unaffected.

(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'photon-tensor)
(require 'nl-llm-attn)
(require 'nl-llm-block)
(require 'nl-llm-decode)

(defvar hd--fail 0)
(defun hd--ck (name ok &optional extra)
  (princ (format "%-52s %s  %s\n" name
                 (if ok "PASS" (progn (setq hd--fail (1+ hd--fail)) "FAIL"))
                 (or extra ""))))

(defun hd--mk (rows cols seed)
  "Deterministic (ROWS x COLS) tensor."
  (let ((v (make-vector (* rows cols) 0.0)) (i 0))
    (while (< i (* rows cols))
      (aset v i (* 0.13 (- (mod (+ (* (1+ i) 7) seed) 11) 5)))
      (setq i (1+ i)))
    (photon-tensor (list rows cols) v)))

(defun hd--maxdiff (a b)
  "Largest absolute elementwise difference between tensors A and B."
  (let ((ad (photon-tensor-data a)) (bd (photon-tensor-data b)) (m 0.0))
    (if (/= (length ad) (length bd))
        1.0e30
      (dotimes (i (length ad))
        (let ((d (abs (- (aref ad i) (aref bd i))))) (when (> d m) (setq m d))))
      m)))

;;; --- an independent reference, written from the architecture not the code ---

(defun hd--rope (vec rowbase nheads hd pos rbase)
  "Rotate NHEADS blocks of width HD at ROWBASE by RoPE at POS.
Written out here so a bug in the library's rotation cannot hide by being
shared with the reference."
  (dotimes (h nheads)
    (let ((base (+ rowbase (* h hd))) (half (/ hd 2)))
      (dotimes (m half)
        (let* ((theta (/ (float pos) (expt rbase (/ (* 2.0 m) (float hd)))))
               (c (cos theta)) (s (sin theta))
               (i0 (+ base (* 2 m))) (i1 (+ base (* 2 m) 1))
               (a0 (aref vec i0)) (a1 (aref vec i1)))
          (aset vec i0 (- (* a0 c) (* a1 s)))
          (aset vec i1 (+ (* a0 s) (* a1 c))))))))

(defun hd--ref-gqa (x layer heads kv-heads hd &optional rope-base)
  "Reference causal GQA over X with query width HEADS*HD and KV width KV-HEADS*HD.
:wq is (heads*hd x dim), :wk/:wv are (kv-heads*hd x dim), :wo is (dim x heads*hd)."
  (let* ((sh (photon-tensor-shape x)) (seq (car sh)) (dim (nth 1 sh))
         (qdim (* heads hd)) (kvdim (* kv-heads hd)) (grp (/ heads kv-heads))
         (rbase (or rope-base 10000.0)) (scale (/ 1.0 (sqrt (float hd))))
         (q (copy-sequence
             (photon-tensor-data (photon-tensor-linear x (plist-get layer :wq)))))
         (k (copy-sequence
             (photon-tensor-data (photon-tensor-linear x (plist-get layer :wk)))))
         (v (photon-tensor-data (photon-tensor-linear x (plist-get layer :wv))))
         (ctx (make-vector (* seq qdim) 0.0)))
    (dotimes (i seq)
      (hd--rope q (* i qdim) heads hd i rbase)
      (hd--rope k (* i kvdim) kv-heads hd i rbase))
    (dotimes (h heads)
      (let ((qc (* h hd)) (kc (* (/ h grp) hd)))
        (dotimes (i seq)
          (let ((scores (make-vector (1+ i) 0.0)) (mx -1.0e30))
            (dotimes (j (1+ i))
              (let ((acc 0.0))
                (dotimes (t0 hd)
                  (setq acc (+ acc (* (aref q (+ (* i qdim) qc t0))
                                      (aref k (+ (* j kvdim) kc t0))))))
                (aset scores j (* acc scale))
                (when (> (aref scores j) mx) (setq mx (aref scores j)))))
            (let ((sm 0.0))
              (dotimes (j (1+ i))
                (aset scores j (exp (- (aref scores j) mx)))
                (setq sm (+ sm (aref scores j))))
              (dotimes (t0 hd)
                (let ((acc 0.0))
                  (dotimes (j (1+ i))
                    (setq acc (+ acc (* (/ (aref scores j) sm)
                                        (aref v (+ (* j kvdim) kc t0))))))
                  (aset ctx (+ (* i qdim) qc t0) acc))))))))
    (photon-tensor-linear (photon-tensor (list seq qdim) ctx)
                          (plist-get layer :wo))))

(defun hd--layer (dim heads kv-heads hd &optional with-key)
  "Build an attention layer plist for the given widths.
WITH-KEY non-nil also records :head-dim, which is how a caller asks for a head
width other than (/ dim heads)."
  (let ((qdim (* heads hd)) (kvdim (* kv-heads hd)))
    (append (when with-key (list :head-dim hd))
            (list :wq (hd--mk qdim dim 1)
                  :wk (hd--mk kvdim dim 2)
                  :wv (hd--mk kvdim dim 3)
                  :wo (hd--mk dim qdim 4)))))

;;; --- checks --------------------------------------------------------------

(let* ((dim 8) (seq 4) (heads 2) (kv-heads 1)
       (x (hd--mk seq dim 9))
       (coupled (/ dim heads))            ; 4  -- the only width that used to work
       (decoupled 6))                     ; 6  -- heads*6 = 12 /= dim, like Qwen3

  ;; 1. The coupled case must be untouched.  This is the regression guard: it
  ;; passes before and after, and compares the library against the independent
  ;; reference rather than against a frozen constant.
  (let* ((layer (hd--layer dim heads kv-heads coupled))
         (got (nl-llm-gqa x layer heads kv-heads))
         (want (hd--ref-gqa x layer heads kv-heads coupled)))
    (hd--ck "coupled head dim == reference" (< (hd--maxdiff got want) 1.0e-12)
            (format "maxdiff %.2e" (hd--maxdiff got want))))

  ;; 2. Stating :head-dim explicitly at the coupled width must change nothing.
  (let* ((plain (hd--layer dim heads kv-heads coupled))
         (keyed (hd--layer dim heads kv-heads coupled t))
         (a (nl-llm-gqa x plain heads kv-heads))
         (b (nl-llm-gqa x keyed heads kv-heads)))
    (hd--ck "explicit :head-dim = dim/heads is a no-op"
            (< (hd--maxdiff a b) 1.0e-12)
            (format "maxdiff %.2e" (hd--maxdiff a b))))

  ;; 3. The decoupled case.  Before Phase 2a this is where the silent wrong
  ;; answer showed up: :head-dim was ignored, (/ dim heads) won, and the loops
  ;; strided 4 through a 12-wide query row.
  (let* ((layer (hd--layer dim heads kv-heads decoupled t))
         (got (nl-llm-gqa x layer heads kv-heads))
         (want (hd--ref-gqa x layer heads kv-heads decoupled)))
    (hd--ck "decoupled head dim == reference" (< (hd--maxdiff got want) 1.0e-12)
            (format "maxdiff %.2e" (hd--maxdiff got want))))

  ;; 4. Output shape stays (seq x dim): :wo folds heads*head-dim back to dim.
  (let* ((layer (hd--layer dim heads kv-heads decoupled t))
         (got (nl-llm-gqa x layer heads kv-heads)))
    (hd--ck "decoupled output shape is (seq x dim)"
            (equal (photon-tensor-shape got) (list seq dim))
            (format "%S" (photon-tensor-shape got))))

  ;; 5. The cached decode path must still equal the full forward.  Both used to
  ;; share the (/ dim heads) assumption, so this agreed while both were wrong;
  ;; it is meaningful only alongside check 3.
  (let* ((layer (hd--layer dim heads kv-heads decoupled t))
         (full (nl-llm-gqa x layer heads kv-heads))
         (cached (nl-llm-gqa-cached x layer heads kv-heads)))
    (hd--ck "decoupled cached == full forward" (< (hd--maxdiff full cached) 1.0e-12)
            (format "maxdiff %.2e" (hd--maxdiff full cached))))

  ;; 6. A KV cache built for a decoupled head dim must size itself by it.
  (let ((cache (nl-llm-kv-new seq dim heads kv-heads decoupled)))
    (hd--ck "kv cache width follows :head-dim"
            (= (length (nl-llm-kv-k cache)) (* seq kv-heads decoupled))
            (format "%d slots, want %d"
                    (length (nl-llm-kv-k cache)) (* seq kv-heads decoupled))))

  ;; 7. Qwen3-0.6B's actual ratio, at a size Elisp can run: query width wider
  ;; than dim is the shape that used to be unrepresentable.
  (let* ((qdim (* heads decoupled)))
    (hd--ck "the shape under test really is Qwen-like" (> qdim dim)
            (format "heads*head-dim = %d > dim = %d" qdim dim))))

;; 8. The whole-model path threads :head-dim from the model plist.
(let* ((dim 8) (heads 2) (kv-heads 1) (hd 6) (vocab 5) (tokens '(0 2 4))
       (blk (append (hd--layer dim heads kv-heads hd t)
                    (list :ln1g (photon-tensor (list dim) (make-vector dim 1.0))
                          :ln2g (photon-tensor (list dim) (make-vector dim 1.0))
                          :wg (hd--mk (* 2 dim) dim 5)
                          :wu (hd--mk (* 2 dim) dim 6)
                          :wd (hd--mk dim (* 2 dim) 7))))
       (model (list :dim dim :heads heads :kv-heads kv-heads :head-dim hd
                    :wte (hd--mk vocab dim 8)
                    :blocks (list blk)
                    :lnf (photon-tensor (list dim) (make-vector dim 1.0))
                    :head (hd--mk vocab dim 10))))
  (condition-case err
      (let ((logits (nl-llm-model-forward model tokens)))
        (hd--ck "model forward runs with a decoupled :head-dim"
                (equal (photon-tensor-shape logits) (list (length tokens) vocab))
                (format "%S" (photon-tensor-shape logits))))
    (error (hd--ck "model forward runs with a decoupled :head-dim" nil
                   (format "%S" err)))))

;;; --- the paths that do NOT implement :head-dim must refuse it ------------
;;
;; Only nl-llm-gqa, nl-llm-block / nl-llm-model-forward and the CPU KV decode
;; honour a decoupled width.  The others still derive it, and a wrong width is
;; silent there, so they call
;; `nl-llm-attn-reject-decoupled-head-dim' instead of ignoring the key.  Each
;; guard is exercised here: a gate nobody has seen go red is not a gate.

(defun hd--raises (thunk pattern)
  "Non-nil when THUNK signals an error whose message matches PATTERN."
  (condition-case err (progn (funcall thunk) nil)
    (error (and (string-match-p pattern (error-message-string err)) t))))

(let* ((dim 8) (heads 2) (kv-heads 1) (hd 6)
       (blk (hd--layer dim heads kv-heads hd t))
       (xrow (hd--mk 1 dim 11)))

  ;; The helper itself: fires on a decoupled width, silent on the derived one.
  (hd--ck "guard fires on a decoupled width"
          (hd--raises (lambda ()
                        (nl-llm-attn-reject-decoupled-head-dim
                         blk dim heads "probe"))
                      "does not implement :head-dim"))
  (hd--ck "guard is silent at the derived width"
          (null (nl-llm-attn-reject-decoupled-head-dim
                 (hd--layer dim heads kv-heads (/ dim heads) t)
                 dim heads "probe")))

  ;; StreamingLLM bounded decode.
  (require 'nl-llm-stream)
  (hd--ck "nl-llm-stream-block refuses :head-dim"
          (hd--raises (lambda ()
                        (nl-llm-stream-block
                         xrow blk (nl-llm-scache-new 1 2 dim heads kv-heads)))
                      "nl-llm-stream-block does not implement"))

  ;; The integrated paged + streaming + ternary block.
  (require 'nl-llm-integrated)
  (hd--ck "nl-llm-integrated--blk refuses :head-dim"
          (hd--raises (lambda ()
                        (nl-llm-integrated--blk
                         xrow blk (nl-llm-spcache-new 1 2 dim heads kv-heads 1)
                         #'ignore))
                      "nl-llm-integrated--blk does not implement"))

  ;; The training path, which Doc 08 flagged as the regression risk.
  (require 'photon-autograd)
  (require 'nl-llm-autograd)
  (hd--ck "nl-llm-ag-block refuses :head-dim"
          (hd--raises (lambda ()
                        (nl-llm-ag-block
                         (photon-autograd-const (hd--mk 2 dim 12))
                         blk heads kv-heads))
                      "nl-llm-ag-block does not implement")))

;; nl-llm-gpu--decode-block carries the same guard, but exercising it needs a
;; Vulkan device, so it is covered by the helper checks above rather than by a
;; call here.  Phase 2b migrates that path instead of guarding it.

(princ (format "\n%s: %d failure(s)\n"
               (if (zerop hd--fail) "head-dim OK" "head-dim") hd--fail))
(when (> hd--fail 0) (kill-emacs 1))
