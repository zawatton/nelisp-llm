;;; agent-unicode-model-test.el --- Unicode native model tests -*- lexical-binding: t; -*-
;; emacs -Q --batch -l test/agent-unicode-model-test.el

(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-agent-model)

(defvar agent-unicode-model-test--fail 0)

(defun agent-unicode-model-test--ck (name ok &optional extra)
  (princ (format "%-64s %s  %s\n" name
                 (if ok "PASS"
                   (setq agent-unicode-model-test--fail
                         (1+ agent-unicode-model-test--fail))
                   "FAIL")
                 (or extra ""))))

(defun agent-unicode-model-test--errors (thunk)
  (condition-case err
      (progn (funcall thunk) nil)
    (error (error-message-string err))))

(defun agent-unicode-model-test--tensor (shape &optional value)
  (let ((size 1))
    (dolist (dimension shape) (setq size (* size dimension)))
    (photon-tensor shape (make-vector size (or value 0.0)))))

(defun agent-unicode-model-test--block (dim ff)
  (list :ln1g (agent-unicode-model-test--tensor (list dim) 1.0)
        :wq (agent-unicode-model-test--tensor (list dim dim))
        :bq (agent-unicode-model-test--tensor (list dim))
        :wk (agent-unicode-model-test--tensor (list dim dim))
        :bk (agent-unicode-model-test--tensor (list dim))
        :wv (agent-unicode-model-test--tensor (list dim dim))
        :bv (agent-unicode-model-test--tensor (list dim))
        :wo (agent-unicode-model-test--tensor (list dim dim))
        :bo (agent-unicode-model-test--tensor (list dim))
        :ln2g (agent-unicode-model-test--tensor (list dim) 1.0)
        :wg (agent-unicode-model-test--tensor (list ff dim))
        :bg (agent-unicode-model-test--tensor (list ff))
        :wu (agent-unicode-model-test--tensor (list ff dim))
        :bu (agent-unicode-model-test--tensor (list ff))
        :wd (agent-unicode-model-test--tensor (list dim ff))
        :bd (agent-unicode-model-test--tensor (list dim))))

(defun agent-unicode-model-test--model (vocab &optional tokenizer)
  (let ((dim 2))
    (list :blocks (list (agent-unicode-model-test--block dim 2))
          :wte (agent-unicode-model-test--tensor (list vocab dim))
          :lnfg (agent-unicode-model-test--tensor (list dim) 1.0)
          :bh (agent-unicode-model-test--tensor (list vocab))
          :dim dim :heads 1 :kvh 1 :vocab vocab
          :tokenizer tokenizer)))

;; Both candidates start E3 81; only the third byte distinguishes them.  The
;; grammar must not observe a partial byte sequence.
(let* ((calls nil)
       (observed nil)
       (position 0)
       (logits (make-vector 256 -1.0))
       (step
        (lambda (id)
          (push id calls)
          (setq position (1+ position))
          (let ((next (make-vector 256 -1.0)))
            (cond ((= position 1) (aset next #x81 9.0))
                  ((= position 2) (aset next #x84 9.0)))
            next)))
       (grammar
        (lambda (emitted)
          (push emitted observed)
          (if (equal emitted "") '(:allow "あい") :stop))))
  (aset logits #xe3 9.0)
  (let ((out (nl-llm-agent-constrained-generate
              logits step grammar "utf8-byte-v1")))
    (agent-unicode-model-test--ck
     "same-prefix Unicode candidates use later byte logits"
     (and (equal out "い")
          (equal (nreverse calls) '(#xe3 #x81 #x84))
          (equal (nreverse observed) '("" "い"))))))

(dolist (case '(("zy" nil "z")
                ("いう" "utf8-byte-v1" "い")))
  (let* ((chars (nth 0 case))
         (tokenizer (nth 1 case))
         (expected (nth 2 case))
         (vocab (nl-llm-agent-tokenizer-vocab tokenizer))
         (logits (make-vector vocab 0.0))
         (first t)
         (out
          (nl-llm-agent-constrained-generate
           logits (lambda (_id) logits)
           (lambda (_emitted)
             (if (prog1 first (setq first nil))
                 (list :allow chars)
               :stop))
           tokenizer)))
    (agent-unicode-model-test--ck
     (format "equal-logit %s tie preserves candidate order"
             (or tokenizer "ASCII"))
     (equal out expected) out)))

(let ((calls nil)
      (seen nil)
      (count 0)
      (logits (make-vector 256 0.0)))
  (let ((out
         (nl-llm-agent-constrained-generate
          logits
          (lambda (id) (push id calls) logits)
          (lambda (emitted)
            (push emitted seen)
            (prog1 (if (= count 0) '(:force ?🙂) :stop)
              (setq count (1+ count))))
          "utf8-byte-v1")))
    (agent-unicode-model-test--ck
     "forced Unicode consumes every byte before the next callback"
     (and (equal out "🙂")
          (equal (nreverse calls) '(#xf0 #x9f #x99 #x82))
          (equal (nreverse seen) '("" "🙂"))))))

(agent-unicode-model-test--ck
 "empty and out-of-range constrained candidates are rejected"
 (and
  (agent-unicode-model-test--errors
   (lambda ()
     (nl-llm-agent-constrained-generate
      (make-vector 256 0.0) (lambda (_id) (make-vector 256 0.0))
      (lambda (_emitted) '(:allow "")) "utf8-byte-v1")))
  (agent-unicode-model-test--errors
   (lambda ()
     (nl-llm-agent-constrained-generate
      (make-vector 96 0.0) (lambda (_id) (make-vector 96 0.0))
      (lambda (_emitted) '(:allow "日")))))))

(let* ((model (agent-unicode-model-test--model 256 "utf8-byte-v1"))
       (messages '((user . "日本語")))
       (character-count (length (nl-llm-agent--render messages)))
       (token-count
        (length
         (nl-llm-agent-tokenizer-encode
          (nl-llm-agent--render messages) "utf8-byte-v1")))
       (cache-calls 0)
       (step-calls 0)
       (policy (nl-llm-agent-model-policy model (lambda (_text) :stop)
                                          character-count))
       (message nil))
  (cl-letf (((symbol-function 'nl-llm-dcache-new)
             (lambda (&rest _args) (setq cache-calls (1+ cache-calls))))
            ((symbol-function 'nl-llm-agent-model-step-fn)
             (lambda (&rest _args)
               (setq step-calls (1+ step-calls))
               (lambda (_id) (make-vector 256 0.0)))))
    (setq message
          (agent-unicode-model-test--errors
           (lambda () (funcall policy messages)))))
  (agent-unicode-model-test--ck
   "UTF-8 prompt overflow is counted in tokens before cache allocation"
   (and (> token-count character-count)
        message (string-match-p "prompt token length" message)
        (= cache-calls 0) (= step-calls 0))
   message))

(agent-unicode-model-test--ck
 "model tokenizer and vocabulary mismatches are rejected"
 (and
  (agent-unicode-model-test--errors
   (lambda ()
     (nl-llm-agent-model-policy
      (agent-unicode-model-test--model 96 "utf8-byte-v1")
      (lambda (_text) :stop))))
  (agent-unicode-model-test--errors
   (lambda ()
     (nl-llm-agent-model-policy
      (agent-unicode-model-test--model 256 nil)
      (lambda (_text) :stop))))))

(let* ((model (agent-unicode-model-test--model 256 "utf8-byte-v1"))
       (policy (nl-llm-agent-model-policy
                model
                (let ((first t))
                  (lambda (_emitted)
                    (if (prog1 first (setq first nil))
                        '(:force ?日)
                      :stop)))
                64))
       (nl-llm-agent-model-inference-mode 'source)
       (out (funcall policy nil)))
  (agent-unicode-model-test--ck
   "actual tiny 256-vocabulary model performs Unicode inference"
   (equal out "日") out))

(princ (format "NL-LLM-AGENT-UNICODE-MODEL %s (%d failures)\n"
               (if (= agent-unicode-model-test--fail 0)
                   "ALL-PASS"
                 "HAS-FAILURES")
               agent-unicode-model-test--fail))
(kill-emacs (if (= agent-unicode-model-test--fail 0) 0 1))

;;; agent-unicode-model-test.el ends here
