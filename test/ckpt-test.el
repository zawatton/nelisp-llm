;;; ckpt-test.el --- checkpoint save/load round-trip  -*- lexical-binding: t; -*-
;; Checks nl-llm-ckpt-save/load round-trips a modern model (config, step, and all
;; tensors incl. block biases) exactly, and rejects a bad format tag.  Pure CPU.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/ckpt-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-ckpt)

(defvar ck--fail 0)
(defun ck--ck (name ok &optional extra)
  (princ (format "%-44s %s  %s\n" name (if ok "PASS" (progn (setq ck--fail (1+ ck--fail)) "FAIL")) (or extra ""))))
(defun ck--t (shape seed) (let ((n 1)) (dolist (d shape) (setq n (* n d)))
  (photon-tensor shape (let ((v (make-vector n 0.0)) (i 0))
    (while (< i n) (aset v i (* 0.123456789 (- (mod (+ (* (1+ i) 7) seed) 97) 48))) (setq i (1+ i))) v))))
(defun ck--teq (a b) (and (equal (photon-tensor-shape a) (photon-tensor-shape b))
                          (let ((da (photon-tensor-data a)) (db (photon-tensor-data b)) (ok t) (i 0))
                            (while (< i (length da)) (unless (= (aref da i) (aref db i)) (setq ok nil)) (setq i (1+ i))) ok)))

(let* ((dim 16) (kvdim 8) (ff 24) (vocab 12)
       (mkblk (lambda (s) (list :ln1g (ck--t (list dim) (+ s 1)) :wq (ck--t (list dim dim) (+ s 2)) :bq (ck--t (list dim) (+ s 3))
                                :wk (ck--t (list kvdim dim) (+ s 4)) :bk (ck--t (list kvdim) (+ s 5))
                                :wv (ck--t (list kvdim dim) (+ s 6)) :bv (ck--t (list kvdim) (+ s 7))
                                :wo (ck--t (list dim dim) (+ s 8)) :bo (ck--t (list dim) (+ s 9)) :ln2g (ck--t (list dim) (+ s 10))
                                :wg (ck--t (list ff dim) (+ s 11)) :bg (ck--t (list ff) (+ s 12))
                                :wu (ck--t (list ff dim) (+ s 13)) :bu (ck--t (list ff) (+ s 14))
                                :wd (ck--t (list dim ff) (+ s 15)) :bd (ck--t (list dim) (+ s 16)))))
       (model (list :config (list :dim dim :heads 4 :kv-heads 2 :ff ff :vocab vocab :nblocks 2)
                    :step 137 :wte (ck--t (list vocab dim) 100) :lnfg (ck--t (list dim) 200) :bh (ck--t (list vocab) 300)
                    :blocks (list (funcall mkblk 1000) (funcall mkblk 2000))))
       (path (make-temp-file "nl-llm-ckpt" nil ".sexp")))
  (nl-llm-ckpt-save path model)
  (let ((m2 (nl-llm-ckpt-load path)))
    (ck--ck "config round-trips" (equal (plist-get m2 :config) (plist-get model :config)))
    (ck--ck "step round-trips" (= (plist-get m2 :step) 137))
    (ck--ck "wte exact" (ck--teq (plist-get m2 :wte) (plist-get model :wte)))
    (ck--ck "lnfg exact" (ck--teq (plist-get m2 :lnfg) (plist-get model :lnfg)))
    (ck--ck "bh exact" (ck--teq (plist-get m2 :bh) (plist-get model :bh)))
    (let ((ok t) (b1 (plist-get model :blocks)) (b2 (plist-get m2 :blocks)))
      (ck--ck "block count" (= (length b1) (length b2)))
      (cl-loop for blk1 in b1 for blk2 in b2 do
        (let ((kv blk1)) (while kv (unless (ck--teq (cadr kv) (plist-get blk2 (car kv))) (setq ok nil)) (setq kv (cddr kv)))))
      (ck--ck "all block tensors (incl. biases) exact" ok)))
  ;; bad format -> error
  (with-temp-file path (prin1 (list :format "bogus" :step 0) (current-buffer)))
  (ck--ck "bad format rejected" (condition-case nil (progn (nl-llm-ckpt-load path) nil) (error t)))
  (delete-file path))

(princ (format "NL-LLM-CKPT %s (%d failures)\n" (if (= ck--fail 0) "ALL-PASS" "HAS-FAILURES") ck--fail))
(kill-emacs (if (= ck--fail 0) 0 1))
;;; ckpt-test.el ends here
