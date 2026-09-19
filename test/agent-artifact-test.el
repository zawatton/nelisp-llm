;;; agent-artifact-test.el --- promoted native artifact catalog tests  -*- lexical-binding: t; -*-

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-agent-artifact)
(require 'nl-llm-agent-improve)
(require 'nl-llm-evolve)

(defvar agent-artifact--fail 0)

(defun agent-artifact--ck (name ok)
  (princ (format "%-67s %s\n" name
                 (if ok "PASS"
                   (setq agent-artifact--fail
                         (1+ agent-artifact--fail))
                   "FAIL"))))

(defun agent-artifact--error-p (thunk)
  (condition-case nil
      (progn (funcall thunk) nil)
    (error t)))

(defun agent-artifact--tensor (shape value)
  (let ((size 1))
    (dolist (dimension shape)
      (setq size (* size dimension)))
    (photon-tensor shape (make-vector size (float value)))))

(defun agent-artifact--model (value)
  (let ((tensor (lambda (shape)
                  (agent-artifact--tensor shape value))))
    (list
     :config '(:dim 2 :heads 1 :kv-heads 1 :ff 2 :vocab 96 :nblocks 1)
     :step 7
     :wte (agent-artifact--tensor '(96 2) value)
     :lnfg (agent-artifact--tensor '(2) value)
     :bh (agent-artifact--tensor '(96) value)
     :blocks
     (list
      (list :ln1g (funcall tensor '(2))
            :wq (funcall tensor '(2 2)) :bq (funcall tensor '(2))
            :wk (funcall tensor '(2 2)) :bk (funcall tensor '(2))
            :wv (funcall tensor '(2 2)) :bv (funcall tensor '(2))
            :wo (funcall tensor '(2 2)) :bo (funcall tensor '(2))
            :ln2g (funcall tensor '(2))
            :wg (funcall tensor '(2 2)) :bg (funcall tensor '(2))
            :wu (funcall tensor '(2 2)) :bu (funcall tensor '(2))
            :wd (funcall tensor '(2 2)) :bd (funcall tensor '(2)))))))

(defun agent-artifact--max-difference (left right)
  (let ((result 0.0))
    (dotimes (index (length left))
      (setq result
            (max result (abs (- (aref left index) (aref right index))))))
    result))

(let* ((trainable (nl-llm-agent-improve-model 4 6 96 1 1))
       (exported (nl-llm-agent-artifact-export-pav trainable 23))
       (tokens '(1 2 3 4))
       (expected (nl-llm-agent--p5-last-logits trainable tokens))
       (caches (list (nl-llm-dcache-new 16 4 1 1)))
       (actual nil)
       (snapshot
        (aref (photon-tensor-data (plist-get exported :wh)) 0)))
  (dolist (token tokens)
    (setq actual
          (nl-llm-decode-step
           token (plist-get exported :blocks) caches
           (plist-get exported :wte) (plist-get exported :lnfg)
           (plist-get exported :bh) 4 nil (plist-get exported :wh))))
  (agent-artifact--ck
   "PAV export preserves the independently trained output head"
   (and (= (plist-get exported :step) 23)
        (< (agent-artifact--max-difference expected actual) 1e-5)))
  (let* ((directory (make-temp-file "nl-llm-pav-resume-" t))
         (catalog-file (expand-file-name "catalog.json" directory)))
    (unwind-protect
        (progn
          (nl-llm-agent-artifact-publish
           catalog-file exported :id "resume-g7" :generation 7 :score 1.25
           :grammar '(:type "done" :length 4 :allow "ab "))
          (let ((loaded
                 (nl-llm-agent-artifact-load-pav
                  catalog-file "resume-g7")))
            (agent-artifact--ck
             "verified artifact reloads as an isolated trainable PAV champion"
             (and (pav-p (plist-get loaded :wh))
                  (= (plist-get loaded :artifact-generation) 7)
                  (= (plist-get loaded :step) 23)
                  (< (agent-artifact--max-difference
                      expected
                      (nl-llm-agent--p5-last-logits loaded tokens))
                     1e-5)))
            (aset
             (photon-tensor-data (pav-value (plist-get loaded :wh)))
             0 99.0)
            (agent-artifact--ck
             "resumed trainable parameters do not mutate immutable artifacts"
             (= (aref (photon-tensor-data (plist-get exported :wh)) 0)
                snapshot))))
      (delete-directory directory t)))
  (aset
   (photon-tensor-data (pav-value (plist-get trainable :wh)))
   0 (+ snapshot 1.0))
  (agent-artifact--ck
   "exported weights remain isolated from later live training"
   (= (aref (photon-tensor-data (plist-get exported :wh)) 0) snapshot)))

(let* ((directory (make-temp-file "nl-llm-artifacts-" t))
       (catalog-file (expand-file-name "catalog.json" directory))
       (grammar '(:type "done" :length 8 :allow "abc "))
       (provider nil)
       (registry nil)
       (models-seen nil))
  (unwind-protect
      (progn
        (let ((descriptor
               (nl-llm-agent-artifact-publish
                catalog-file (agent-artifact--model 0.1)
                :id "champion-g1" :name "Champion generation 1"
                :grammar grammar :maxseq 256 :score 0.7 :generation 1)))
          (agent-artifact--ck
           "promotion writes one public descriptor with measured provenance"
           (and (equal (plist-get descriptor :id) "champion-g1")
                (= (plist-get descriptor :score) 0.7)
                (= (plist-get descriptor :generation) 1)
                (= (length (plist-get descriptor :sha256)) 64))))
        (agent-artifact--ck
         "catalog and immutable checkpoint are private files"
         (and (= (logand (file-modes catalog-file) #o777) #o600)
              (= (logand
                  (file-modes
                   (expand-file-name
                    "champion-g1-g000001.sexp" directory))
                  #o777)
                 #o600)))
        (agent-artifact--ck
         "catalog exposes no checkpoint path or live model object"
         (let ((public
                (car (nl-llm-agent-artifact-catalog catalog-file))))
           (and (not (plist-member public :checkpoint))
                (not (plist-member public :checkpoint-path))
                (not (plist-member public :model))
                (equal (plist-get public :capabilities)
                       '(generate local trained promoted
                         constrained-decoding)))))
        (setq provider
              (nl-llm-agent-artifact-provider
               "native" catalog-file "Native champions"))
        (setq registry (nl-llm-agent-provider-registry-new))
        (nl-llm-agent-provider-register registry provider)
        (cl-letf
            (((symbol-function 'nl-llm-agent-model-policy)
              (lambda (model grammar maxseq)
                (setq models-seen
                      (append models-seen (list (list model grammar maxseq))))
                (lambda (messages)
                  (format "native:%d" (length messages))))))
          (let* ((session
                  (nl-llm-agent-session-open
                   registry "native/champion-g1"))
                 (reply
                  (nl-llm-agent-session-complete
                   session '((user . "hello")))))
            (agent-artifact--ck
             "provider verifies and opens a promoted checkpoint lazily"
             (and (equal reply "native:1")
                  (= (length models-seen) 1)
                  (= (nth 2 (car models-seen)) 256)))))
        (nl-llm-agent-artifact-publish
         catalog-file (agent-artifact--model 0.2)
         :id "champion-g2" :name "Champion generation 2"
         :grammar grammar :maxseq 512 :score 0.9 :generation 2)
        (agent-artifact--ck
         "provider discovers a later promotion without reconstruction"
         (equal
          (mapcar
           (lambda (item) (plist-get item :qualified-id))
           (nl-llm-agent-provider-models registry))
          '("native/champion-g1" "native/champion-g2")))
        (let ((before
               (prin1-to-string
                (nl-llm-agent-artifact-catalog catalog-file))))
          (agent-artifact--ck
           "duplicate artifact ids leave the active catalog unchanged"
           (and
            (agent-artifact--error-p
             (lambda ()
               (nl-llm-agent-artifact-publish
                catalog-file (agent-artifact--model 0.3)
                :id "champion-g2" :grammar grammar
                :score 1.0 :generation 2)))
            (equal before
                   (prin1-to-string
                    (nl-llm-agent-artifact-catalog catalog-file))))))
        (let* ((candidate-model
                (nl-llm-agent-improve-model 2 2 96 1 1))
               (score
                (lambda (candidate)
                  (aref
                   (photon-tensor-data
                    (pav-value (plist-get candidate :bh)))
                   0)))
               (state
                (nl-llm-evolution-new
                 candidate-model score
                 :publish
                 (nl-llm-agent-artifact-publisher
                  catalog-file :id-prefix "evolved" :grammar grammar
                  :export
                  (lambda (candidate)
                    (nl-llm-agent-artifact-export-pav candidate 1)))))
               (result
                (nl-llm-evolution-step
                 state
                 (lambda (candidate _state)
                   (aset
                    (photon-tensor-data
                     (pav-value (plist-get candidate :bh)))
                    0 1.1)))))
          (agent-artifact--ck
           "evolution promotion commits only after immutable publication"
           (and
            (eq (plist-get result :status) 'promoted)
            (= (nl-llm-evolution-generation state) 1)
            (equal
             (mapcar
              (lambda (item) (plist-get item :id))
              (nl-llm-agent-artifact-catalog catalog-file))
             '("champion-g1" "champion-g2" "evolved-g1")))))
        (let* ((session
                (nl-llm-agent-session-open registry "native/evolved-g1"))
               (reply
                (nl-llm-agent-session-complete
                 session '((user . "exercise the exported head")))))
          (agent-artifact--ck
           "published PAV generation performs real constrained inference"
           (and (string-prefix-p "DONE " reply)
                (= (length reply) 13))))
        (write-region
         " " nil
         (expand-file-name "champion-g1-g000001.sexp" directory)
         t 'silent)
        (agent-artifact--ck
         "digest mismatch blocks a modified checkpoint before inference"
         (agent-artifact--error-p
          (lambda ()
            (nl-llm-agent-session-open
             registry "native/champion-g1"))))
        (agent-artifact--ck
         "data grammar rejects code-like unsupported implementations"
         (agent-artifact--error-p
          (lambda ()
            (nl-llm-agent-artifact--grammar-spec
             '(:type "elisp" :function "eval") "test grammar")))))
    (delete-directory directory t)))

(princ (format "NL-LLM-AGENT-ARTIFACT %s (%d failures)\n"
               (if (= agent-artifact--fail 0)
                   "ALL-PASS"
                 "HAS-FAILURES")
               agent-artifact--fail))
(kill-emacs (if (= agent-artifact--fail 0) 0 1))

;;; agent-artifact-test.el ends here
