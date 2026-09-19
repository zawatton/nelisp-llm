;;; decode-capacity-test.el --- bounded native context validation  -*- lexical-binding: t; -*-
;; emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/decode-capacity-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-decode)
(require 'nl-llm-agent-model)

(defvar dcap--fail 0)
(defun dcap--ck (name ok &optional extra)
  (princ (format "%-58s %s  %s\n" name
                 (if ok "PASS" (progn (setq dcap--fail (1+ dcap--fail)) "FAIL"))
                 (or extra ""))))

(defun dcap--error-message (thunk)
  (condition-case err
      (progn (funcall thunk) nil)
    (error (error-message-string err))))

(defun dcap--zero-tensor (shape)
  (let ((size 1))
    (dolist (n shape) (setq size (* size n)))
    (photon-tensor shape (make-vector size 0.0))))

(defun dcap--block (dim heads kvh)
  (let* ((hd (/ dim heads)) (kvdim (* kvh hd)) (ff dim))
    (list :ln1g (dcap--zero-tensor (list dim))
          :wq (dcap--zero-tensor (list dim dim)) :bq (dcap--zero-tensor (list dim))
          :wk (dcap--zero-tensor (list kvdim dim)) :bk (dcap--zero-tensor (list kvdim))
          :wv (dcap--zero-tensor (list kvdim dim)) :bv (dcap--zero-tensor (list kvdim))
          :wo (dcap--zero-tensor (list dim dim)) :bo (dcap--zero-tensor (list dim))
          :ln2g (dcap--zero-tensor (list dim))
          :wg (dcap--zero-tensor (list ff dim)) :bg (dcap--zero-tensor (list ff))
          :wu (dcap--zero-tensor (list ff dim)) :bu (dcap--zero-tensor (list ff))
          :wd (dcap--zero-tensor (list dim ff)) :bd (dcap--zero-tensor (list dim)))))

(let ((bad '((0 4 2 1) (-1 4 2 1) (1 0 2 1) (1 4 0 1) (1 4 2 0)
             (1.5 4 2 1) (1 5 2 1) (1 4 3 2))))
  (dolist (args bad)
    (dcap--ck (format "invalid cache dimensions rejected: %S" args)
              (dcap--error-message (lambda () (apply #'nl-llm-dcache-new args))))))

(let ((cache (nl-llm-dcache-new 1 3 1 1)))
  (dcap--ck "positive odd head dimension is accepted"
            (and (= (nl-llm-dcache-kvdim cache) 3)
                 (= (length (nl-llm-dcache-k cache)) 3)
                 (= (length (nl-llm-dcache-v cache)) 3))))

(dolist (bad-len '(-1 1.5))
  (let ((cache (nl-llm-dcache-new 2 4 2 1)))
    (setf (nl-llm-dcache-len cache) bad-len)
    (let ((msg (dcap--error-message
                (lambda () (nl-llm-decode-block nil nil cache)))))
      (dcap--ck (format "malformed cache len rejected: %S" bad-len)
                (and msg (string-match-p "non-negative integer" msg)) msg))))

(let ((cache (nl-llm-dcache-new 2 4 2 1)))
  (setf (nl-llm-dcache-kvdim cache) 1)
  (let ((msg (dcap--error-message
              (lambda () (nl-llm-decode-block nil nil cache)))))
    (dcap--ck "malformed cache kvdim rejected"
              (and msg (string-match-p "does not match expected width" msg)) msg)))

(let ((cache (nl-llm-dcache-new 2 4 2 1)))
  (setf (nl-llm-dcache-heads cache) 3)
  (let ((msg (dcap--error-message
              (lambda () (nl-llm-decode-block nil nil cache)))))
    (dcap--ck "malformed cache divisibility rejected"
              (and msg (string-match-p "must be divisible" msg)) msg)))

(let ((cache (nl-llm-dcache-new 2 4 2 1)))
  (let ((msg (dcap--error-message
              (lambda () (nl-llm-decode-h 0 '(nil) (list cache)
                                           nil nil 8)))))
    (dcap--ck "cache/decoder dim mismatch rejected"
              (and msg (string-match-p "does not match decoder dim" msg)) msg)))

(let* ((dim 4) (heads 2) (kvh 1) (vocab 3)
       (block (dcap--block dim heads kvh))
       (cache (nl-llm-dcache-new 1 dim heads kvh))
       (wte (dcap--zero-tensor (list vocab dim)))
       (lnfg (dcap--zero-tensor (list dim)))
       (bh (dcap--zero-tensor (list vocab))))
  (nl-llm-decode-step 0 (list block) (list cache) wte lnfg bh dim)
  (dcap--ck "exactly-full cache accepts its final token"
            (= (nl-llm-dcache-len cache) 1))
  (let ((msg (dcap--error-message
              (lambda () (nl-llm-decode-step 0 (list block) (list cache)
                                              wte lnfg bh dim)))))
    (dcap--ck "overflow reports context capacity"
              (and msg (string-match-p "context capacity" msg)) msg)))

(let ((msg (dcap--error-message
            (lambda () (nl-llm-decode-step 0 '(nil nil)
                                            (list (nl-llm-dcache-new 1 4 2 1))
                                            nil nil nil 4)))))
  (dcap--ck "block/cache count mismatch rejected before decode"
            (and msg (string-match-p "count mismatch" msg)) msg))

(let* ((c1 (nl-llm-dcache-new 2 4 2 1))
       (c2 (nl-llm-dcache-new 1 4 2 1))
       (k1 (copy-sequence (nl-llm-dcache-k c1)))
       (v1 (copy-sequence (nl-llm-dcache-v c1))))
  (setf (nl-llm-dcache-len c2) 1)
  (let ((msg (dcap--error-message
              (lambda () (nl-llm-decode-step 0 '(nil nil) (list c1 c2)
                                              nil nil nil 4)))))
    (dcap--ck "multi-block overflow is detected atomically"
              (and msg
                   (string-match-p "context capacity" msg)
                   (= (nl-llm-dcache-len c1) 0)
                   (equal (nl-llm-dcache-k c1) k1)
                   (equal (nl-llm-dcache-v c1) v1)) msg)))

(let* ((c1 (nl-llm-dcache-new 2 4 2 1))
       (c2 (nl-llm-dcache-new 2 4 2 1))
       (k1 (copy-sequence (nl-llm-dcache-k c1)))
       (v1 (copy-sequence (nl-llm-dcache-v c1)))
       (k2 (copy-sequence (nl-llm-dcache-k c2)))
       (v2 (copy-sequence (nl-llm-dcache-v c2))))
  (setf (nl-llm-dcache-len c2) 1)
  (let ((msg (dcap--error-message
              (lambda () (nl-llm-decode-step 0 '(nil nil) (list c1 c2)
                                              nil nil nil 4)))))
    (dcap--ck "different cache lengths are rejected atomically"
              (and msg
                   (string-match-p "cache length mismatch" msg)
                   (= (nl-llm-dcache-len c1) 0)
                   (= (nl-llm-dcache-len c2) 1)
                   (equal (nl-llm-dcache-k c1) k1)
                   (equal (nl-llm-dcache-v c1) v1)
                   (equal (nl-llm-dcache-k c2) k2)
                   (equal (nl-llm-dcache-v c2) v2)) msg)))

(let* ((c1 (nl-llm-dcache-new 2 4 2 1))
       (c2 (nl-llm-dcache-new 1 4 2 1))
       (k1 (copy-sequence (nl-llm-dcache-k c1)))
       (v1 (copy-sequence (nl-llm-dcache-v c1)))
       (k2 (copy-sequence (nl-llm-dcache-k c2)))
       (v2 (copy-sequence (nl-llm-dcache-v c2))))
  (setf (nl-llm-dcache-len c2) 1)
  (let ((msg (dcap--error-message
              (lambda () (nl-llm-decode-h 0 '(nil nil) (list c1 c2)
                                           nil nil 4)))))
    (dcap--ck "decode-h multi-block overflow leaves every cache unchanged"
              (and msg
                   (string-match-p "context capacity" msg)
                   (= (nl-llm-dcache-len c1) 0)
                   (= (nl-llm-dcache-len c2) 1)
                   (equal (nl-llm-dcache-k c1) k1)
                   (equal (nl-llm-dcache-v c1) v1)
                   (equal (nl-llm-dcache-k c2) k2)
                   (equal (nl-llm-dcache-v c2) v2)) msg)))

(let ((cache (nl-llm-dcache-new 1 4 2 1)))
  (setf (nl-llm-dcache-len cache) 1)
  (let ((msg (dcap--error-message
              (lambda () (nl-llm-decode-block nil nil cache)))))
    (dcap--ck "direct decode-block call guards capacity before append"
              (and msg (string-match-p "context capacity" msg)) msg)))

(let ((cache (nl-llm-dcache-new 2 4 2 1)))
  (setf (nl-llm-dcache-v cache) (make-vector 3 0.0))
  (let ((msg (dcap--error-message
              (lambda () (nl-llm-decode-h 0 '(nil) (list cache) nil nil 4)))))
    (dcap--ck "inconsistent cache vector sizes are rejected"
              (and msg (string-match-p "vector sizes differ" msg)) msg)))

(let* ((calls-new 0) (calls-step 0)
       (policy (nl-llm-agent-model-policy
                '(:blocks (nil) :dim 4 :heads 2 :kvh 1)
                (lambda (_emitted) :stop) 3))
       msg)
  (cl-letf (((symbol-function 'nl-llm-dcache-new)
             (lambda (&rest _args) (setq calls-new (1+ calls-new)) nil))
            ((symbol-function 'nl-llm-agent-model-step-fn)
             (lambda (&rest _args) (setq calls-step (1+ calls-step)) nil)))
    (setq msg (dcap--error-message (lambda () (funcall policy '((user . "long")))))))
  (dcap--ck "policy rejects prompt overflow before allocation or model step"
            (and msg (string-match-p "prompt length" msg)
                 (= calls-new 0) (= calls-step 0)) msg))

(let* ((messages '((user . "ok")))
       (capacity (length (nl-llm-agent--render messages)))
       (calls-new 0) (calls-step 0) (calls-token 0)
       (policy (nl-llm-agent-model-policy
                '(:blocks (nil) :dim 4 :heads 2 :kvh 1)
                (lambda (_emitted) :stop) capacity))
       first second)
  (cl-letf (((symbol-function 'nl-llm-dcache-new)
             (lambda (&rest _args) (setq calls-new (1+ calls-new)) 'cache))
            ((symbol-function 'nl-llm-agent-model-step-fn)
             (lambda (&rest _args)
               (setq calls-step (1+ calls-step))
               (lambda (_id)
                 (setq calls-token (1+ calls-token))
                 (make-vector nl-llm-agent-char-vocab 0.0)))))
    (setq first (funcall policy messages))
    (setq second (funcall policy messages)))
  (dcap--ck "exact-capacity stop policy succeeds with fresh caches"
            (and (equal first "") (equal second "")
                 (= calls-new 2) (= calls-step 2)
                 (= calls-token (* 2 capacity)))))

(princ (format "NL-LLM-DECODE-CAPACITY %s (%d failures)\n"
               (if (= dcap--fail 0) "ALL-PASS" "HAS-FAILURES") dcap--fail))
(kill-emacs (if (= dcap--fail 0) 0 1))
;;; decode-capacity-test.el ends here
