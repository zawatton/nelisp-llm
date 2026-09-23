;;; student-soft-test.el --- student soft-target training checks  -*- lexical-binding: t; -*-

(setq load-prefer-newer t)
(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'cl-lib)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-agent-improve)
(require 'nl-llm-soft-loss)
(require 'nl-llm-student)

(defvar sst--fail 0)
(defvar sst--pass 0)
(defun sst--check (name ok &optional detail)
  (if ok
      (setq sst--pass (1+ sst--pass))
    (setq sst--fail (1+ sst--fail)))
  (princ (format "%-64s %s%s\n" name
                 (if ok "PASS" "FAIL")
                 (if detail (concat "  " detail) ""))))
(defun sst--scalar (value)
  (aref (photon-tensor-data (pav-value value)) 0))
(defun sst--finite-p (x)
  (and (= x x) (< (abs x) 1.0e308)))
(defun sst--row-targets (rows)
  (if (vectorp rows) (copy-sequence rows) (apply #'vector rows)))
(defun sst--soft-value (data vocab targets mask)
  (let ((row-targets (sst--row-targets targets)))
    (photon-autograd-reset-tape)
    (sst--scalar
     (nl-llm-ag-soft-kl
      (photon-autograd-const
       (photon-tensor (list (length row-targets) vocab)
                      (apply #'vector data)))
      row-targets (copy-sequence mask)))))

;; A donor-sized vocabulary is accepted without making the tokenizer's size a
;; model invariant.  Keep the hidden state small; this checks the seam only.
(let* ((model (nl-llm-student-new :dim 2 :ff 2 :blocks 1 :heads 1
                                  :vocab 151936 :tokenizer 'qwen-byte))
       (logits (nl-llm-agent--p5-forward model '(1 2 3)))
       (data (photon-tensor-data (pav-value logits))))
  (sst--check "student accepts donor vocabulary and forwards"
              (= (length data) (* 3 151936))
              (format "logits=%d" (length data))))

;; Compare every trainable component against a central difference through the
;; actual P5 tape.  The objective is deliberately small but exercises all
;; parameters of a block, embedding, norm, head, and bias.
(let* ((model (nl-llm-student-new :dim 8 :ff 8 :blocks 1 :heads 1 :vocab 16))
       (parameters (nl-llm-agent--p5-params model))
       (tokens '(1 2 3 4))
       (targets (vector '((1 . -0.2) (2 . -1.0))
                        '((2 . -0.1) (3 . -1.2))
                        '((3 . -0.3) (4 . -0.9))
                        '((4 . -0.4) (5 . -0.8))))
       (mask [1 1 1 1])
       (eps 1.0e-5)
       (logits (nl-llm-agent--p5-forward model tokens))
       (loss (nl-llm-ag-soft-kl logits targets mask))
       (worst 0.0) (components 0))
  (photon-autograd-zero-grad parameters)
  (photon-autograd-backward loss)
  (dolist (parameter parameters)
    (let* ((values (photon-tensor-data (pav-value parameter)))
           (gradient (photon-tensor-data (pav-grad parameter))))
      (dotimes (i (length values))
        (let ((original (aref values i)))
          (aset values i (+ original eps))
          (let ((plus (sst--scalar
                       (nl-llm-ag-soft-kl
                        (nl-llm-agent--p5-forward model tokens)
                        targets mask))))
            (aset values i (- original eps))
            (let* ((minus (sst--scalar
                           (nl-llm-ag-soft-kl
                            (nl-llm-agent--p5-forward model tokens)
                            targets mask)))
                   (numerical (/ (- plus minus) (* 2.0 eps)))
                   (analytic (aref gradient i))
                   (relative (/ (abs (- numerical analytic))
                                (max 1.0e-8 (abs numerical) (abs analytic)))))
              (setq worst (max worst relative))
              (setq components (1+ components))))
          (aset values i original)))))
  (sst--check "soft KL tape matches finite differences for every parameter"
              (< worst 1.0e-5)
              (format "components=%d worst=%.3g" components worst))
  ;; A factor-of-two analytic gradient must not pass the same check.
  (let ((rejected 0))
    (dolist (parameter parameters)
      (let* ((values (photon-tensor-data (pav-value parameter)))
             (gradient (photon-tensor-data (pav-grad parameter))))
        (dotimes (i (length values))
          (let ((original (aref values i)))
            (aset values i (+ original eps))
            (let ((plus (sst--scalar
                         (nl-llm-ag-soft-kl
                          (nl-llm-agent--p5-forward model tokens)
                          targets mask))))
              (aset values i (- original eps))
              (let* ((minus (sst--scalar
                             (nl-llm-ag-soft-kl
                              (nl-llm-agent--p5-forward model tokens)
                              targets mask)))
                     (numerical (/ (- plus minus) (* 2.0 eps)))
                     (relative (/ (abs (- numerical (* 2.0 (aref gradient i))))
                                  (max 1.0e-8 (abs numerical)
                                        (abs (* 2.0 (aref gradient i)))))))
                (when (>= relative 1.0e-5) (setq rejected (1+ rejected))))
              (aset values i original))))))
    (sst--check "factor-two gradient control is rejected"
                (> rejected 0) (format "rejected=%d" rejected))))

;; Nil targets and masked rows are absent from both value and gradient.
(let* ((data '(0.2 -0.3 0.7 1.1 -0.4 0.3 0.6 -0.8 0.5))
       (targets (vector '((0 . -0.2) (1 . -0.7)) nil
                        '((2 . -0.1) (1 . -1.0))))
       (all [1 1 1])
       (without (sst--soft-value (append (cl-subseq data 0 3)
                                         (cl-subseq data 6 9))
                                 3 (vector (aref targets 0)
                                           (aref targets 2)) [1 1]))
       (with-nil (sst--soft-value data 3 targets all)))
  (sst--check "nil target row contributes nothing"
              (< (abs (- with-nil without)) 1.0e-12)
              (format "with=%.12g without=%.12g" with-nil without))
  (let* ((logits (photon-autograd-const (photon-tensor '(3 3) (vconcat data))))
         (loss (nl-llm-ag-soft-kl logits targets all)))
    (photon-autograd-backward loss)
    (let ((gradient (photon-tensor-data (pav-grad logits))))
      (sst--check "nil target row has exact zero gradient"
                  (and (= (aref gradient 3) 0.0)
                       (= (aref gradient 4) 0.0)
                       (= (aref gradient 5) 0.0)))))
  (let ((real (vector (aref targets 0) '((1 . -0.1) (2 . -0.8))
                       (aref targets 2))))
    (sst--check "real target in formerly nil row changes value"
                (> (abs (- (sst--soft-value data 3 real all) with-nil))
                   1.0e-10)))
  (sst--check "masked row contributes nothing"
              (< (abs (- (sst--soft-value data 3 targets [1 1 0])
                         (sst--soft-value (cl-subseq data 0 6)
                                          3 (vector (aref targets 0)
                                                    (aref targets 1)) [1 1])))
                 1.0e-12))
  (let* ((logits (photon-autograd-const (photon-tensor '(3 3) (vconcat data))))
         (loss (nl-llm-ag-soft-kl logits targets [1 1 0])))
    (photon-autograd-backward loss)
    (let ((gradient (photon-tensor-data (pav-grad logits))))
      (sst--check "masked target row has exact zero gradient"
                  (and (= (aref gradient 6) 0.0)
                       (= (aref gradient 7) 0.0)
                       (= (aref gradient 8) 0.0))))))

;; End-to-end completion-only training, including a dropped unresolvable
;; sampled token in the report.
(let* ((table (make-hash-table :test #'equal))
       (surface (lambda (char) (nl-llm-token-table-key (list char)))))
  (puthash (funcall surface ?a) 1 table)
  (puthash (funcall surface ?b) 2 table)
  (let* ((tokenize (lambda (text)
                     (mapcar (lambda (c) (if (= c ?p) 0 (1- (- c ?a))))
                             (string-to-list text))))
         (position (lambda (char)
                     (list :token (char-to-string char) :bytes (list char)
                           :top (list (list :token (char-to-string char)
                                             :bytes (list char) :logprob 0.0)
                                      (list :token (if (= char ?a) "b" "a")
                                            :bytes (list (if (= char ?a) ?b ?a))
                                            :logprob -0.5)))))
         (examples (vector
                    (list :prompt "p" :completion "a"
                          :tokens (list (funcall position ?a)))
                    (list :prompt "p" :completion "b"
                          :tokens (list (funcall position ?b)))
                    (list :prompt "p" :completion "a"
                          :tokens (list (funcall position ?z)))))
         (model (nl-llm-student-new :dim 4 :ff 4 :blocks 1 :heads 1 :vocab 8))
         (report (nl-llm-soft-train model examples :lr 0.1 :epochs 3
                                    :tokenize tokenize :table table)))
    (sst--check "soft training reduces loss"
                (< (plist-get report :loss-after)
                   (plist-get report :loss-before))
                (format "before=%.8g after=%.8g"
                        (plist-get report :loss-before)
                        (plist-get report :loss-after)))
    (sst--check "unresolvable sampled position is visible in report"
                (< (plist-get report :positions-with-targets)
                   (plist-get report :positions))
                (format "with=%d positions=%d"
                        (plist-get report :positions-with-targets)
                        (plist-get report :positions))))
  (let* ((model (nl-llm-student-new :dim 4 :ff 4 :blocks 1 :heads 1 :vocab 8))
         (report (nl-llm-soft-train model
                                    (vector (list :prompt "p" :completion "a"
                                                  :tokens (list (list :token "a"
                                                                          :bytes '(97)
                                                                       :top (list (list :token "a"
                                                                                        :bytes '(97)
                                                                                        :logprob 0.0)
                                                                                  (list :token "b"
                                                                                        :bytes '(98)
                                                                                        :logprob -0.5))))))
                                    :lr 0.0 :epochs 1
                                    :tokenize (lambda (text)
                                                (mapcar (lambda (c)
                                                          (if (= c ?p) 0
                                                            (1- (- c ?a))))
                                                        (string-to-list text)))
                                    :table table)))
    (sst--check "zero learning rate leaves loss unchanged"
                (= (plist-get report :loss-before)
                   (plist-get report :loss-after))
                (format "before=%.12g after=%.12g"
                        (plist-get report :loss-before)
                        (plist-get report :loss-after)))))

(princ (format "\nstudent soft: %d passed, %d failed\n" sst--pass sst--fail))
(kill-emacs (if (= sst--fail 0) 0 1))
;;; student-soft-test.el ends here
