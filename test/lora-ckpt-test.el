;;; lora-ckpt-test.el --- LoRA adapter checkpoint round-trip  -*- lexical-binding: t; -*-
;; Checks standalone LoRA adapter checkpoint save/load, validation, description,
;; size claims, and merge semantics.  Pure CPU.
;;   emacs -Q --batch -l test/lora-ckpt-test.el
(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-ckpt)
(require 'nl-llm-lora-ckpt)

(defvar lck--fail 0)

(defun lck--ck (name ok &optional extra)
  (princ
   (format "%-46s %s  %s\n"
           name
           (if ok
               "PASS"
             (setq lck--fail (1+ lck--fail))
             "FAIL")
           (or extra ""))))

(defun lck--t (shape seed)
  (let ((n 1))
    (dolist (d shape)
      (setq n (* n d)))
    (photon-tensor
     shape
     (let ((v (make-vector n 0.0))
           (i 0))
       (while (< i n)
         (aset v i (* 0.0703125 (- (mod (+ (* (1+ i) 11) seed) 113) 56)))
         (setq i (1+ i)))
       v))))

(defun lck--teq (a b)
  (and (equal (photon-tensor-shape a) (photon-tensor-shape b))
       (let ((da (photon-tensor-data a))
             (db (photon-tensor-data b))
             (ok t)
             (i 0))
         (while (< i (length da))
           (unless (= (aref da i) (aref db i))
             (setq ok nil))
           (setq i (1+ i)))
         ok)))

(defun lck--adapter (rank in out seed &optional alpha)
  (list :a (lck--t (list rank in) (+ seed 1))
        :b (lck--t (list out rank) (+ seed 2))
        :rank rank
        :alpha (or alpha (* 2 rank))
        :in in
        :out out))

(defun lck--adapter-equal (a b)
  (and (= (plist-get a :rank) (plist-get b :rank))
       (= (plist-get a :alpha) (plist-get b :alpha))
       (= (plist-get a :in) (plist-get b :in))
       (= (plist-get a :out) (plist-get b :out))
       (lck--teq (plist-get a :a) (plist-get b :a))
       (lck--teq (plist-get a :b) (plist-get b :b))))

(defun lck--adapters-equal (a b)
  (and (= (length a) (length b))
       (let ((ok t)
             (rest a))
         (while rest
           (let* ((site (caar rest))
                  (adapter (cdar rest))
                  (other (assoc site b)))
             (unless (and other (lck--adapter-equal adapter (cdr other)))
               (setq ok nil)))
           (setq rest (cdr rest)))
         ok)))

(defun lck--copy (obj)
  (let ((print-length nil)
        (print-level nil))
    (read (prin1-to-string obj))))

(defun lck--read-form (path)
  (with-temp-buffer
    (let ((coding-system-for-read 'utf-8))
      (insert-file-contents path))
    (goto-char (point-min))
    (read (buffer-string))))

(defun lck--write-form (path form)
  (let ((text (let ((print-length nil)
                    (print-level nil))
                (prin1-to-string form))))
    (let ((coding-system-for-write 'utf-8))
      (write-region text nil path nil 'silent))))

(defun lck--file-size (path)
  (nth 7 (file-attributes path)))

(defun lck--make-temp-file (prefix &optional suffix)
  (let ((temporary-file-directory "/tmp/"))
    (make-temp-file prefix nil suffix)))

(defun lck--describe-totals (text)
  (when (string-match
         "Total: adapter floats \\([0-9]+\\), full-weight floats \\([0-9]+\\), ratio \\([0-9.]+\\)%\\'"
         text)
    (list (string-to-number (match-string 1 text))
          (string-to-number (match-string 2 text))
          (string-to-number (match-string 3 text)))))

(defun lck--adapter-float-total (adapters)
  (let ((n 0))
    (dolist (cell adapters n)
      (let ((adapter (cdr cell)))
        (setq n (+ n (* (plist-get adapter :rank)
                        (+ (plist-get adapter :in)
                           (plist-get adapter :out)))))))))

(defun lck--full-float-total (adapters)
  (let ((n 0))
    (dolist (cell adapters n)
      (let ((adapter (cdr cell)))
        (setq n (+ n (* (plist-get adapter :in)
                        (plist-get adapter :out))))))))

(defun lck--size-t (shape seed)
  "Return a constant tensor for the large checkpoint size fixture."
  (let ((n 1))
    (dolist (dim shape)
      (setq n (* n dim)))
    (photon-tensor shape
                   (make-vector n (* 0.015625 (1+ (mod seed 17)))))))

(defun lck--size-adapter (rank in out seed)
  "Return a constant-valued adapter for the checkpoint size fixture."
  (list :a (lck--size-t (list rank in) (+ seed 1))
        :b (lck--size-t (list out rank) (+ seed 2))
        :rank rank
        :alpha (* 2 rank)
        :in in
        :out out))

(defun lck--mk-block (dim kvdim ff seed)
  (list :ln1g (lck--size-t (list dim) (+ seed 1))
        :wq   (lck--size-t (list dim dim) (+ seed 2))
        :bq   (lck--size-t (list dim) (+ seed 3))
        :wk   (lck--size-t (list kvdim dim) (+ seed 4))
        :bk   (lck--size-t (list kvdim) (+ seed 5))
        :wv   (lck--size-t (list kvdim dim) (+ seed 6))
        :bv   (lck--size-t (list kvdim) (+ seed 7))
        :wo   (lck--size-t (list dim dim) (+ seed 8))
        :bo   (lck--size-t (list dim) (+ seed 9))
        :ln2g (lck--size-t (list dim) (+ seed 10))
        :wg   (lck--size-t (list ff dim) (+ seed 11))
        :bg   (lck--size-t (list ff) (+ seed 12))
        :wu   (lck--size-t (list ff dim) (+ seed 13))
        :bu   (lck--size-t (list ff) (+ seed 14))
        :wd   (lck--size-t (list dim ff) (+ seed 15))
        :bd   (lck--size-t (list dim) (+ seed 16))))

(defun lck--size-model ()
  (let ((dim 36)
        (kvdim 18)
        (ff 72)
        (vocab 18))
    (list :config (list :dim dim :heads 6 :kv-heads 3 :ff ff :vocab vocab :nblocks 1)
          :step 23
          :wte (lck--size-t (list vocab dim) 100)
          :lnfg (lck--size-t (list dim) 200)
          :bh (lck--size-t (list vocab) 300)
          :blocks (list (lck--mk-block dim kvdim ff 1000)))))

(defun lck--size-adapters ()
  (list
   (cons :block-0-wq (lck--size-adapter 1 36 36 500))
   (cons :block-0-wv (lck--size-adapter 1 36 18 600))
   (cons :block-0-wo (lck--size-adapter 1 36 36 700))
   (cons :block-0-wg (lck--size-adapter 1 36 72 800))
   (cons :block-0-wu (lck--size-adapter 1 36 72 900))
   (cons :block-0-wd (lck--size-adapter 1 72 36 1000))))

;; exact round-trip + meta round-trip
(let* ((adapters (list (cons :block-0-wq (lck--adapter 2 6 8 10 16))
                       (cons "block-0-wv" (lck--adapter 3 5 7 20 24))))
       (meta (list :step 91
                   :base-config (list :dim 8 :heads 2 :kv-heads 1 :ff 16)
                   :note "round-trip"))
       (path (lck--make-temp-file "nl-llm-lora-ckpt" ".sexp")))
  (unwind-protect
      (progn
        (nl-llm-lora-ckpt-save path adapters meta)
        (let ((ckpt (nl-llm-lora-ckpt-load path)))
          (lck--ck "format round-trips"
                   (equal (plist-get ckpt :format) nl-llm-lora-ckpt-format))
          (lck--ck "adapters exact round-trip"
                   (lck--adapters-equal adapters (plist-get ckpt :adapters)))
          (lck--ck "meta round-trips"
                   (equal (plist-get ckpt :meta) meta))))
    (when (file-exists-p path)
      (delete-file path))))

;; bad tag rejected
(let ((path (lck--make-temp-file "nl-llm-lora-bad-tag" ".sexp")))
  (unwind-protect
      (progn
        (lck--write-form path (list :format "nl-llm-ckpt-v1" :meta nil :adapters nil))
        (let ((msg (condition-case err
                       (progn (nl-llm-lora-ckpt-load path) nil)
                     (error (error-message-string err)))))
          (lck--ck "bad tag rejected"
                   (and msg
                        (string-match-p "nl-llm-ckpt-v1" msg)
                        (string-match-p "nl-llm-lora-ckpt-v1" msg)))))
    (when (file-exists-p path)
      (delete-file path))))

;; corrupt adapter rejected on load
(let* ((site :broken-site)
       (adapters (list (cons site (lck--adapter 2 5 7 30))))
       (path (lck--make-temp-file "nl-llm-lora-corrupt" ".sexp")))
  (unwind-protect
      (progn
        (nl-llm-lora-ckpt-save path adapters)
        (let* ((form (lck--read-form path))
               (adapter (cdr (assoc site (plist-get form :adapters))))
               (a (plist-get adapter :a)))
          (aset a 0 (list 9 9))
          (lck--write-form path form))
        (let ((msg (condition-case err
                       (progn (nl-llm-lora-ckpt-load path) nil)
                     (error (error-message-string err)))))
          (lck--ck "corrupt adapter rejected on load"
                   (and msg (string-match-p ":broken-site" msg))
                   msg)))
    (when (file-exists-p path)
      (delete-file path))))

;; malformed adapter rejected on save
(let* ((site :bad-save)
       (bad (lck--adapter 2 5 7 40))
       (path (lck--make-temp-file "nl-llm-lora-bad-save" ".sexp")))
  (unwind-protect
      (progn
        (delete-file path)
        (aset (plist-get bad :a) 0 (list 4 5))
        (let ((msg (condition-case err
                       (progn (nl-llm-lora-ckpt-save path (list (cons site bad))) nil)
                     (error (error-message-string err)))))
          (lck--ck "malformed adapter rejected on save"
                   (and msg
                        (string-match-p ":bad-save" msg)
                        (not (file-exists-p path)))
                   (or msg ""))))
    (when (file-exists-p path)
      (delete-file path))))

;; size claim is real
(let* ((adapters (lck--size-adapters))
       (model (lck--size-model))
       (expected-adapter (lck--adapter-float-total adapters))
       (expected-full (lck--full-float-total adapters))
       (adapter-path (lck--make-temp-file "nl-llm-lora-size" ".sexp"))
       (full-path (lck--make-temp-file "nl-llm-full-size" ".sexp")))
  (unwind-protect
      (progn
        (nl-llm-lora-ckpt-save adapter-path adapters
                               (list :step 23 :base-config (plist-get model :config)))
        (nl-llm-ckpt-save full-path model)
        (let* ((desc (nl-llm-lora-ckpt-describe adapter-path))
               (totals (lck--describe-totals desc))
               (adapter-total (nth 0 totals))
               (full-total (nth 1 totals))
               (ratio (nth 2 totals))
               (adapter-bytes (lck--file-size adapter-path))
               (full-bytes (lck--file-size full-path))
               (byte-ratio (* 100.0 (/ (float adapter-bytes) full-bytes))))
          (lck--ck "describe totals accurate"
                   (and totals
                        (= adapter-total expected-adapter)
                        (= full-total expected-full))
                   (format "%d/%d" adapter-total full-total))
          (lck--ck "size ratio under 5%"
                   (< ratio 5.0)
                   (format "%.4f%% (%d/%d)" ratio adapter-total full-total))
          (lck--ck "adapter file smaller than full ckpt"
                   (< adapter-bytes full-bytes)
                   (format "%d < %d (%.4f%%)" adapter-bytes full-bytes byte-ratio))))
    (when (file-exists-p adapter-path)
      (delete-file adapter-path))
    (when (file-exists-p full-path)
      (delete-file full-path))))

;; merge-plists
(let* ((left (list (cons :shared (lck--adapter 2 4 6 50 10))
                   (cons :left-only (lck--adapter 1 3 5 60 7))))
       (right (list (cons :shared (lck--adapter 2 4 6 70 12))
                    (cons "right-only" (lck--adapter 1 5 4 80 9))))
       (left0 (lck--copy left))
       (right0 (lck--copy right))
       (merged (nl-llm-lora-ckpt-merge-plists left right)))
  (lck--ck "merge right-hand side wins"
           (lck--adapter-equal (cdr (assoc :shared merged))
                               (cdr (assoc :shared right))))
  (lck--ck "merge keeps left unique sites"
           (lck--adapter-equal (cdr (assoc :left-only merged))
                               (cdr (assoc :left-only left))))
  (lck--ck "merge keeps right unique sites"
           (lck--adapter-equal (cdr (assoc "right-only" merged))
                               (cdr (assoc "right-only" right))))
  (lck--ck "merge does not mutate inputs"
           (and (equal left left0) (equal right right0))))

;; Contract lock: `nl-llm-lora-make' accepts any numeric ALPHA, so a fractional
;; alpha (e.g. alpha = rank/2) must survive save/load rather than being refused
;; by the checkpoint validator.
(let* ((adapters (list (cons :frac-alpha (lck--adapter 2 6 8 20 3.0))))
       (path (lck--make-temp-file "nl-llm-lora-frac-alpha" ".sexp")))
  (unwind-protect
      (let (saved)
        (setq saved (condition-case e
                        (progn (nl-llm-lora-ckpt-save path adapters) t)
                      (error (format "%s" (error-message-string e)))))
        (lck--ck "fractional alpha saves" (eq saved t)
                 (if (eq saved t) "" saved))
        (when (eq saved t)
          (let ((back (nl-llm-lora-ckpt-load path)))
            (lck--ck "fractional alpha round-trips"
                     (equal (plist-get (cdr (assoc :frac-alpha
                                                   (plist-get back :adapters)))
                                       :alpha)
                            3.0))
            (lck--ck "describe renders fractional alpha"
                     (string-match-p "alpha 3\\.0"
                                     (nl-llm-lora-ckpt-describe back))))))
    (when (file-exists-p path) (delete-file path))))

(princ (format "NL-LLM-LORA-CKPT %s (%d failures)\n"
               (if (= lck--fail 0) "ALL-PASS" "HAS-FAILURES")
               lck--fail))
(kill-emacs (if (= lck--fail 0) 0 1))
;;; lora-ckpt-test.el ends here
