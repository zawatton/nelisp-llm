;;; attn-test.el --- KV-cache / GQA attention equivalence tests  -*- lexical-binding: t; -*-
;; Run: emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/attn-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'photon-tensor)
(require 'nl-llm-attn)

(defvar at--fail 0)
(defun at--ck (name ok &optional extra)
  (princ (format "%-46s %s  %s\n" name (if ok "PASS"
                                         (progn (setq at--fail (1+ at--fail)) "FAIL"))
                 (or extra ""))))
(defun at--mk (rows cols seed)
  (let ((v (make-vector (* rows cols) 0.0)) (i 0))
    (while (< i (* rows cols))
      (aset v i (* 0.1 (- (mod (+ (* (1+ i) 7) seed) 11) 5))) (setq i (1+ i)))
    (photon-tensor (list rows cols) v)))
(defun at--maxdiff (ta tb)
  (let* ((a (photon-tensor-data ta)) (b (photon-tensor-data tb))
         (n (length a)) (m 0.0) (i 0))
    (while (< i n)
      (let ((e (abs (- (aref a i) (aref b i))))) (when (> e m) (setq m e)))
      (setq i (1+ i)))
    m))

;; MHA (kv-heads = heads): cached KV decode == full recomputation
(let* ((dim 8) (heads 2) (seq 5)
       (layer (list :wq (at--mk dim dim 1) :wk (at--mk dim dim 2)
                    :wv (at--mk dim dim 3) :wo (at--mk dim dim 4)))
       (x (at--mk seq dim 9))
       (d (at--maxdiff (nl-llm-mha x layer heads) (nl-llm-mha-cached x layer heads))))
  (at--ck "mha: kv-cache decode == full" (< d 1.0e-9) (format "maxdiff=%.2e" d)))

;; GQA (heads=4, kv-heads=2): smaller KV, cached == full
(let* ((dim 8) (heads 4) (kvh 2) (seq 6) (hd (/ dim heads)) (kvdim (* kvh hd))
       (layer (list :wq (at--mk dim dim 1) :wk (at--mk kvdim dim 2)
                    :wv (at--mk kvdim dim 3) :wo (at--mk dim dim 4)))
       (x (at--mk seq dim 9))
       (full (nl-llm-gqa x layer heads kvh))
       (cached (nl-llm-gqa-cached x layer heads kvh))
       (d (at--maxdiff full cached)))
  (at--ck "gqa: kv-cache decode == full" (< d 1.0e-9) (format "maxdiff=%.2e" d))
  (at--ck "gqa: output shape (seq x dim)"
          (equal (photon-tensor-shape cached) (list seq dim)))
  (let ((cache (nl-llm-kv-new seq dim heads kvh)))
    (at--ck "gqa: KV cache width < dim (saving)"
            (< (length (nl-llm-kv-k cache)) (* seq dim))
            (format "kvwidth=%d dimwidth=%d" (length (nl-llm-kv-k cache)) (* seq dim)))))

;; cache length grows by one per step
(let* ((dim 8) (heads 2) (seq 4)
       (layer (list :wq (at--mk dim dim 1) :wk (at--mk dim dim 2)
                    :wv (at--mk dim dim 3) :wo (at--mk dim dim 4)))
       (cache (nl-llm-kv-new seq dim heads)) (ok t))
  (dotimes (i seq)
    (nl-llm-attn-step (at--mk 1 dim (+ 20 i)) layer cache)
    (unless (= (nl-llm-kv-len cache) (1+ i)) (setq ok nil)))
  (at--ck "cache length grows per step" ok))

(princ (format "NL-LLM-ATTN %s (%d failures)\n"
               (if (= at--fail 0) "ALL-PASS" "HAS-FAILURES") at--fail))
(kill-emacs (if (= at--fail 0) 0 1))
;;; attn-test.el ends here
