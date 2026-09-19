;;; nl-llm-agent-recur-artifact.el --- persistent recurrent-depth artifacts -*- lexical-binding: t; -*-

;;; Commentary:
;; A bounded, data-only interchange format for recurrent-depth models.  It is
;; deliberately separate from training checkpoints: imported models contain
;; fresh zero gradients, but no optimizer state or resume semantics.

;;; Code:

(require 'cl-lib)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-recur)
(require 'nl-llm-agent-tokenizer)
(require 'nl-llm-agent-recur-supervised)

;; These variables control the reader and must be dynamically bound while the
;; untrusted artifact form is read.  Declaring them special also keeps the
;; strict byte-compiler from treating the bindings as unused lexical locals.
(defvar read-eval)
(defvar read-circle)

(defconst nl-llm-agent-recur-artifact-format
  "nl-llm-recur-artifact-v1")
(defconst nl-llm-agent-recur-artifact-family 'recurrent-depth)
(defconst nl-llm-agent-recur-artifact-max-bytes (* 16 1024 1024))
(defconst nl-llm-agent-recur-artifact-max-scalars 1000000)
(defconst nl-llm-agent-recur-artifact--keys
  '(:format :family :tokenizer :r :s0-seed :step :model))
(defconst nl-llm-agent-recur-artifact--model-keys
  '(:wte :prelude :wa :ba :core :coda :lnfg :wh :bh :dim :heads
    :kv-heads :sigma :vocab :tokenizer :ff))
(defconst nl-llm-agent-recur-artifact--block-keys
  '(:ln1g :wq :bq :wk :bk :wv :bv :wo :bo :ln2g :wg :bg :wu :bu :wd :bd))

(defun nl-llm-agent-recur-artifact--proper-list-p (value)
  "Return non-nil for a finite proper list, detecting circular tails."
  (nl-llm-agent-recur-supervised--proper-list-p value))

(defun nl-llm-agent-recur-artifact--keys (value allowed where)
  "Validate a proper plist VALUE against ALLOWED for WHERE."
  (unless (nl-llm-agent-recur-artifact--proper-list-p value)
    (error "%s must be a proper plist" where))
  (let ((tail value) (seen nil))
    (while tail
      (let ((key (pop tail)))
        (unless tail
          (error "%s has an unpaired key %S" where key))
        (unless (memq key allowed)
          (error "%s has unknown key %S" where key))
        (when (memq key seen)
          (error "%s has duplicate key %S" where key))
        (push key seen)
        (pop tail))))
  value)

(defun nl-llm-agent-recur-artifact--required (value keys where)
  "Require every key in KEYS to occur in plist VALUE for WHERE."
  (dolist (key keys)
    (unless (plist-member value key)
      (error "%s is missing key %S" where key)))
  value)

(defun nl-llm-agent-recur-artifact--finite-p (value)
  "Return non-nil when VALUE is a finite numeric scalar."
  (nl-llm-agent-recur-supervised--finite-p value))

(defun nl-llm-agent-recur-artifact--raw-tensor
    (value where total)
  "Validate raw tensor VALUE and return its scalar count.
TOTAL is the count already accepted in the containing model."
  (unless (and (vectorp value) (= (length value) 2))
    (error "%s must be a raw [shape data] vector" where))
  (let ((shape (aref value 0))
        (data (aref value 1)))
    (unless (and (nl-llm-agent-recur-artifact--proper-list-p shape)
                 (memq (length shape) '(1 2))
                 (cl-every (lambda (dimension)
                             (and (integerp dimension) (> dimension 0)))
                           shape))
      (error "%s has an invalid raw tensor shape" where))
    (let ((size (apply #'* shape)))
      (when (> (+ total size) nl-llm-agent-recur-artifact-max-scalars)
        (error "recurrent artifact exceeds its scalar bound"))
      (unless (and (vectorp data) (= (length data) size))
        (error "%s has an invalid raw tensor data length" where))
      (dotimes (index size)
        (unless (nl-llm-agent-recur-artifact--finite-p (aref data index))
          (error "%s contains a non-finite raw tensor value" where)))
      size)))

(defun nl-llm-agent-recur-artifact--raw-block (block dim ff heads kv-heads total)
  "Validate raw BLOCK and return its cumulative scalar count."
  (nl-llm-agent-recur-artifact--keys
   block nl-llm-agent-recur-artifact--block-keys "raw recurrent block")
  (let ((kvdim (* kv-heads (/ dim heads))))
    (dolist (key nl-llm-agent-recur-artifact--block-keys)
      (let ((shape
             (pcase key
               ((or :ln1g :ln2g) (list dim))
               (:wq (list dim dim)) (:bq (list dim))
               (:wk (list kvdim dim)) (:bk (list kvdim))
               (:wv (list kvdim dim)) (:bv (list kvdim))
               (:wo (list dim dim)) (:bo (list dim))
               (:wg (list ff dim)) (:bg (list ff))
               (:wu (list ff dim)) (:bu (list ff))
               (:wd (list dim ff)) (:bd (list dim)))))
        (setq total
              (+ total
                 (let ((raw (plist-get block key)))
                   (unless (and (vectorp raw) (= (length raw) 2)
                                (equal (aref raw 0) shape))
                     (error "raw recurrent block %S has wrong shape" key))
                   (nl-llm-agent-recur-artifact--raw-tensor
                    raw (format "raw recurrent block %S" key) total)))))
      (when (> total nl-llm-agent-recur-artifact-max-scalars)
        (error "recurrent artifact exceeds its scalar bound")))
    total))

(defun nl-llm-agent-recur-artifact--validate-raw-model (model tokenizer)
  "Validate raw MODEL completely before any PAV allocation.
Return its geometry and total scalar count."
  (nl-llm-agent-recur-artifact--keys
   model nl-llm-agent-recur-artifact--model-keys "raw recurrent model")
  (nl-llm-agent-recur-artifact--required
   model '(:wte :prelude :wa :ba :core :coda :lnfg :wh :bh :dim :heads
           :kv-heads :sigma :vocab :tokenizer :ff)
   "raw recurrent model")
  (let* ((dim (plist-get model :dim))
         (heads (plist-get model :heads))
         (kv-heads (plist-get model :kv-heads))
         (sigma (plist-get model :sigma))
         (vocab (plist-get model :vocab))
         (ff (plist-get model :ff))
         (wte (plist-get model :wte))
         (total 0))
    (unless (and (integerp dim) (> dim 0) (integerp heads) (> heads 0)
                 (= (% dim heads) 0) (cl-evenp (/ dim heads))
                 (integerp kv-heads) (> kv-heads 0)
                 (= (% heads kv-heads) 0)
                 (integerp vocab) (> vocab 0) (integerp ff) (> ff 0)
                 (nl-llm-agent-recur-artifact--finite-p sigma)
                 (>= sigma 0.0))
      (error "raw recurrent model has invalid geometry"))
    (unless (equal (nl-llm-agent-tokenizer-id (plist-get model :tokenizer))
                   tokenizer)
      (error "raw recurrent model tokenizer disagrees"))
    (unless (= vocab (nl-llm-agent-tokenizer-vocab tokenizer))
      (error "raw recurrent model vocab disagrees with tokenizer"))
    (let* ((wte-shape (and (vectorp wte) (= (length wte) 2)
                           (aref wte 0))))
      (unless (equal wte-shape (list vocab dim))
        (error "raw recurrent model wte shape disagrees")))
    (setq total
          (+ total (nl-llm-agent-recur-artifact--raw-tensor
                    wte "raw wte" total)))
    (dolist (stack (list (plist-get model :prelude)
                         (plist-get model :core)
                         (plist-get model :coda)))
      (unless (nl-llm-agent-recur-artifact--proper-list-p stack)
        (error "raw recurrent model block stack must be a proper list")))
    (unless (consp (plist-get model :core))
      (error "raw recurrent model core must be nonempty"))
    (dolist (stack (list (plist-get model :prelude)
                         (plist-get model :core)
                         (plist-get model :coda)))
      (dolist (block stack)
        (unless (nl-llm-agent-recur-artifact--proper-list-p block)
          (error "raw recurrent block must be a proper plist"))
        (setq total
              (nl-llm-agent-recur-artifact--raw-block
               block dim ff heads kv-heads total))))
    (dolist (entry `((:wa ,(list dim (* 2 dim)))
                     (:ba ,(list dim)) (:lnfg ,(list dim))
                     (:wh ,(list vocab dim)) (:bh ,(list vocab))))
      (let ((raw (plist-get model (car entry)))
            (shape (cadr entry)))
        (unless (and (vectorp raw) (= (length raw) 2)
                     (equal (aref raw 0) shape))
          (error "raw recurrent model %S has wrong shape" (car entry)))
        (setq total
              (+ total
                 (nl-llm-agent-recur-artifact--raw-tensor
                  raw (format "raw recurrent model %S" (car entry)) total)))))
    (list :dim dim :heads heads :kv-heads kv-heads :sigma sigma
          :vocab vocab :ff ff :total-scalars total)))

(defun nl-llm-agent-recur-artifact--raw-tensor-from-pav (value)
  "Return a detached raw tensor for PAV VALUE."
  (let ((tensor (pav-value value)))
    (vector (copy-sequence (photon-tensor-shape tensor))
            (copy-sequence (photon-tensor-data tensor)))))

(defun nl-llm-agent-recur-artifact--raw-block-from-pav (block)
  "Return a detached raw BLOCK plist."
  (let (result)
    (dolist (key nl-llm-agent-recur-artifact--block-keys)
      (setq result
            (append result
                    (list key
                          (nl-llm-agent-recur-artifact--raw-tensor-from-pav
                           (plist-get block key))))))
    result))

(defun nl-llm-agent-recur-artifact--raw-model (model geometry tokenizer)
  "Convert validated PAV MODEL to canonical detached raw data."
  (let ((blocks (lambda (stack)
                  (mapcar #'nl-llm-agent-recur-artifact--raw-block-from-pav
                          stack))))
    (list :wte (nl-llm-agent-recur-artifact--raw-tensor-from-pav
                (plist-get model :wte))
          :prelude (funcall blocks (plist-get model :prelude))
          :wa (nl-llm-agent-recur-artifact--raw-tensor-from-pav
               (plist-get model :wa))
          :ba (nl-llm-agent-recur-artifact--raw-tensor-from-pav
               (plist-get model :ba))
          :core (funcall blocks (plist-get model :core))
          :coda (funcall blocks (plist-get model :coda))
          :lnfg (nl-llm-agent-recur-artifact--raw-tensor-from-pav
                 (plist-get model :lnfg))
          :wh (nl-llm-agent-recur-artifact--raw-tensor-from-pav
               (plist-get model :wh))
          :bh (nl-llm-agent-recur-artifact--raw-tensor-from-pav
               (plist-get model :bh))
          :dim (plist-get geometry :dim)
          :heads (plist-get geometry :heads)
          :kv-heads (plist-get geometry :kv-heads)
          :sigma (plist-get geometry :sigma)
          :vocab (plist-get geometry :vocab)
          :tokenizer (copy-sequence tokenizer)
          :ff (plist-get geometry :ff))))

(defun nl-llm-agent-recur-artifact--raw-model-to-pav (model)
  "Convert a completely validated raw MODEL to fresh PAV parameters."
  (let ((blocks (lambda (stack)
                  (mapcar
                   (lambda (block)
                     (let (result)
                       (dolist (key nl-llm-agent-recur-artifact--block-keys)
                         (setq result
                               (append result
                                       (list key
                                             (photon-autograd-const
                                              (photon-tensor
                                               (copy-sequence (aref (plist-get block key) 0))
                                               (copy-sequence (aref (plist-get block key) 1))))))))
                       result))
                   stack))))
    (list :wte (photon-autograd-const
                (photon-tensor (copy-sequence (aref (plist-get model :wte) 0))
                               (copy-sequence (aref (plist-get model :wte) 1))))
          :prelude (funcall blocks (plist-get model :prelude))
          :wa (photon-autograd-const
               (photon-tensor (copy-sequence (aref (plist-get model :wa) 0))
                              (copy-sequence (aref (plist-get model :wa) 1))))
          :ba (photon-autograd-const
               (photon-tensor (copy-sequence (aref (plist-get model :ba) 0))
                              (copy-sequence (aref (plist-get model :ba) 1))))
          :core (funcall blocks (plist-get model :core))
          :coda (funcall blocks (plist-get model :coda))
          :lnfg (photon-autograd-const
                 (photon-tensor (copy-sequence (aref (plist-get model :lnfg) 0))
                                (copy-sequence (aref (plist-get model :lnfg) 1))))
          :wh (photon-autograd-const
               (photon-tensor (copy-sequence (aref (plist-get model :wh) 0))
                              (copy-sequence (aref (plist-get model :wh) 1))))
          :bh (photon-autograd-const
               (photon-tensor (copy-sequence (aref (plist-get model :bh) 0))
                              (copy-sequence (aref (plist-get model :bh) 1))))
          :dim (plist-get model :dim) :heads (plist-get model :heads)
          :kv-heads (plist-get model :kv-heads) :sigma (plist-get model :sigma)
          :vocab (plist-get model :vocab)
          :tokenizer (copy-sequence (plist-get model :tokenizer))
          :ff (plist-get model :ff))))

(defun nl-llm-agent-recur-artifact--options (keys)
  "Validate export KEYS and return canonical options."
  (nl-llm-agent-recur-artifact--keys
   keys '(:tokenizer :r :s0-seed :step) "recurrent artifact options")
  (let* ((tokenizer-value (if (plist-member keys :tokenizer)
                              (plist-get keys :tokenizer)
                            nl-llm-agent-tokenizer-utf8))
         (tokenizer (and (stringp tokenizer-value)
                         (nl-llm-agent-tokenizer-id tokenizer-value)))
         (r (if (plist-member keys :r) (plist-get keys :r) 2))
         (seed (if (plist-member keys :s0-seed)
                   (plist-get keys :s0-seed) 1))
         (step (if (plist-member keys :step) (plist-get keys :step) 0)))
    (unless tokenizer
      (error "recurrent artifact tokenizer must be a supported string"))
    (unless (and (integerp r) (<= 1 r) (<= r 32))
      (error "recurrent artifact r must be an integer in 1..32"))
    (unless (and (integerp seed) (<= 0 seed) (<= seed #xffffffff))
      (error "recurrent artifact s0-seed must be a uint32"))
    (unless (and (integerp step) (>= step 0))
      (error "recurrent artifact step must be a nonnegative integer"))
    (list :tokenizer tokenizer :r r :s0-seed seed :step step)))

;;;###autoload
(cl-defun nl-llm-agent-recur-artifact-export (model &rest keys)
  "Export validated recurrent MODEL as a detached data-only artifact.

The returned S-expression contains no PAV gradients, closures, optimizer
state, or resume claim.  TOKENIZER, R, S0-SEED, and STEP are optional keys."
  (let* ((options (nl-llm-agent-recur-artifact--options keys))
         (geometry
          (nl-llm-agent-recur-supervised--model
           model (plist-get options :tokenizer)))
         (raw (nl-llm-agent-recur-artifact--raw-model
               model geometry (plist-get options :tokenizer))))
    ;; Validate the serialized representation as well.  This keeps the scalar
    ;; bound and raw tensor contract identical for export and import.
    (nl-llm-agent-recur-artifact--validate-raw-model
     raw (plist-get options :tokenizer))
    (list :format (copy-sequence nl-llm-agent-recur-artifact-format)
          :family nl-llm-agent-recur-artifact-family
          :tokenizer (copy-sequence (plist-get options :tokenizer))
          :r (plist-get options :r)
          :s0-seed (plist-get options :s0-seed)
          :step (plist-get options :step)
          :model raw)))

;;;###autoload
(defun nl-llm-agent-recur-artifact-import (document)
  "Import DOCUMENT after raw validation, returning a fresh PAV model bundle."
  (nl-llm-agent-recur-artifact--keys
   document nl-llm-agent-recur-artifact--keys "recurrent artifact")
  (nl-llm-agent-recur-artifact--required
   document nl-llm-agent-recur-artifact--keys "recurrent artifact")
  (let* ((format (plist-get document :format))
         (family (plist-get document :family))
         (tokenizer-value (plist-get document :tokenizer))
         (tokenizer (and (stringp tokenizer-value)
                         (nl-llm-agent-tokenizer-id tokenizer-value)))
         (r (plist-get document :r))
         (seed (plist-get document :s0-seed))
         (step (plist-get document :step))
         (raw-model (plist-get document :model)))
    (unless (equal format nl-llm-agent-recur-artifact-format)
      (error "unsupported recurrent artifact format %S" format))
    (unless (eq family nl-llm-agent-recur-artifact-family)
      (error "unsupported recurrent artifact family %S" family))
    (unless tokenizer
      (error "recurrent artifact tokenizer must be a supported string"))
    (unless (and (integerp r) (<= 1 r) (<= r 32))
      (error "recurrent artifact r must be an integer in 1..32"))
    (unless (and (integerp seed) (<= 0 seed) (<= seed #xffffffff))
      (error "recurrent artifact s0-seed must be a uint32"))
    (unless (and (integerp step) (>= step 0))
      (error "recurrent artifact step must be a nonnegative integer"))
    ;; This is deliberately before every photon-autograd-const call.
    (nl-llm-agent-recur-artifact--validate-raw-model raw-model tokenizer)
    (let ((model (nl-llm-agent-recur-artifact--raw-model-to-pav raw-model)))
      ;; Reuse the production validator after raw validation as a second shape
      ;; and tokenizer check on the freshly allocated representation.
      (nl-llm-agent-recur-supervised--model model tokenizer)
      (list :model model :tokenizer (copy-sequence tokenizer)
            :r r :s0-seed seed :step step
            :family nl-llm-agent-recur-artifact-family))))

(defun nl-llm-agent-recur-artifact--path (path)
  "Return a validated local expanded PATH."
  (unless (and (stringp path) (not (string-empty-p path)))
    (error "recurrent artifact path must be a non-empty string"))
  (let ((expanded (expand-file-name path)))
    (when (file-remote-p expanded)
      (error "recurrent artifact path must be local"))
    expanded))

(defun nl-llm-agent-recur-artifact--serialize (document)
  "Serialize DOCUMENT to bounded UTF-8 bytes with stable float printing."
  (let ((print-length nil) (print-level nil) (print-circle nil)
        (print-escape-nonascii t) (float-output-format nil))
    (encode-coding-string
     (concat (prin1-to-string document) "\n") 'utf-8)))

(defun nl-llm-agent-recur-artifact--read-buffer (path)
  "Read PATH once into a bounded unibyte buffer and return its BYTES."
  (when (> (file-attribute-size (file-attributes path))
           nl-llm-agent-recur-artifact-max-bytes)
    (error "recurrent artifact exceeds its byte bound"))
  (with-temp-buffer
    (set-buffer-multibyte nil)
    (insert-file-contents-literally path nil 0
                                    (1+ nl-llm-agent-recur-artifact-max-bytes))
    (when (> (buffer-size) nl-llm-agent-recur-artifact-max-bytes)
      (error "recurrent artifact exceeds its byte bound"))
    (buffer-string)))

(defun nl-llm-agent-recur-artifact--parse-bytes (bytes)
  "Parse bounded UTF-8 BYTES with evaluation and circular syntax disabled."
  (with-temp-buffer
    (set-buffer-multibyte nil)
    (insert bytes)
    (decode-coding-region (point-min) (point-max) 'utf-8)
    (goto-char (point-min))
    (let ((read-eval nil) (read-circle nil))
      (let ((form (read (current-buffer))))
        (when (re-search-forward "[^[:space:]]" nil t)
          (error "recurrent artifact contains trailing data"))
        form))))

;;;###autoload
(cl-defun nl-llm-agent-recur-artifact-save (path model &rest keys)
  "Atomically save MODEL to local PATH without overwriting an existing file.

Return `(:path EXPANDED :sha256 FILE-DIGEST)'.  All model and option
validation occurs before the target path is touched."
  (let* ((expanded (nl-llm-agent-recur-artifact--path path))
         (document (apply #'nl-llm-agent-recur-artifact-export model keys))
         (bytes (nl-llm-agent-recur-artifact--serialize document))
         (digest (secure-hash 'sha256 bytes))
         (directory (file-name-directory expanded))
         (temporary nil))
    (when (> (length bytes) nl-llm-agent-recur-artifact-max-bytes)
      (error "recurrent artifact exceeds its byte bound"))
    (unless (file-directory-p directory)
      (error "recurrent artifact parent directory does not exist"))
    (when (or (file-exists-p expanded) (file-symlink-p expanded)
              (file-directory-p expanded))
      (error "recurrent artifact refuses to overwrite target"))
    (unwind-protect
        (progn
          (setq temporary
                (make-temp-file
                 (expand-file-name ".nl-llm-recur-artifact-" directory)))
          (let ((coding-system-for-write 'binary))
            (write-region bytes nil temporary nil 'silent))
          ;; nil means rename-file will fail rather than replace a race-created
          ;; target.  The temporary file is in the target's directory.
          (rename-file temporary expanded nil)
          (setq temporary nil)
          (list :path expanded :sha256 digest))
      (when (and temporary (file-exists-p temporary))
        (delete-file temporary)))))

;;;###autoload
(defun nl-llm-agent-recur-artifact-load (path expected-sha256)
  "Load PATH once, checking its literal SHA-256 EXPECTED-SHA256 first.

The hash and parser consume the same bounded byte buffer, avoiding a second
filesystem read between integrity verification and S-expression parsing."
  (let ((expanded (nl-llm-agent-recur-artifact--path path)))
    (unless (and (file-regular-p expanded)
                 (not (file-directory-p expanded)))
      (error "recurrent artifact path must name a regular file"))
    (unless (and (stringp expected-sha256)
                 (string-match-p
                  (rx string-start (= 64 (in "a-f0-9")) string-end)
                  expected-sha256))
      (error "recurrent artifact expected SHA-256 must be lowercase hex"))
    (let* ((bytes (nl-llm-agent-recur-artifact--read-buffer expanded))
           (digest (secure-hash 'sha256 bytes)))
      (unless (equal digest expected-sha256)
        (error "recurrent artifact SHA-256 mismatch"))
      ;; Parse only after the digest matches; both operations use this exact
      ;; bounded byte string, avoiding a second filesystem read or TOCTOU gap.
      (let ((bundle
             (nl-llm-agent-recur-artifact-import
              (nl-llm-agent-recur-artifact--parse-bytes bytes))))
        (plist-put bundle :sha256 digest)))))

(provide 'nl-llm-agent-recur-artifact)
;;; nl-llm-agent-recur-artifact.el ends here
