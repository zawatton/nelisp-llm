;;; nl-llm-agent-artifact.el --- promoted native model artifacts  -*- lexical-binding: t; -*-

;; A promoted checkpoint is immutable.  A small JSON catalog publishes only
;; data and a SHA-256 digest; loading revalidates the digest before model state
;; becomes reachable through the provider boundary.

;;; Code:

(require 'cl-lib)
(require 'json)
(require 'subr-x)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-ckpt)
(require 'nl-llm-agent-tokenizer)
(require 'nl-llm-agent-action-grammar)
(require 'nl-llm-agent-model)
(require 'nl-llm-agent-provider)

(defconst nl-llm-agent-artifact-catalog-format
  "nl-llm-agent-artifact-catalog-v1"
  "Current promoted native model catalog format.")

(defconst nl-llm-agent-artifact-max-catalog-bytes (* 1024 1024)
  "Maximum promoted model catalog size.")

(defconst nl-llm-agent-artifact-default-allow
  "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?-_"
  "Default variable characters for artifact action grammars.")

(defun nl-llm-agent-artifact--keys (value allowed where)
  "Validate plist VALUE against ALLOWED keys for WHERE."
  (unless (and (listp value) (= (% (length value) 2) 0))
    (error "%s must be a JSON object" where))
  (let ((tail value))
    (while tail
      (unless (memq (car tail) allowed)
        (error "%s contains unknown key %S" where (car tail)))
      (setq tail (cddr tail))))
  value)

(defun nl-llm-agent-artifact--id (value where)
  "Return validated artifact identifier VALUE for WHERE."
  (unless (and (stringp value)
               (<= 1 (length value))
               (<= (length value) 128)
               (string-match-p "\\`[A-Za-z0-9_.-]+\\'" value))
    (error "%s has invalid artifact id %S" where value))
  value)

(defun nl-llm-agent-artifact--number (value where)
  "Return finite numeric VALUE for WHERE."
  (unless (and (numberp value) (= value value))
    (error "%s must be a finite number" where))
  value)

(defun nl-llm-agent-artifact--file-actions-spec (value where)
  "Validate a file-actions grammar VALUE for WHERE and return a deep copy."
  (unless (listp value)
    (error "%s must be a JSON object" where))
  (let ((tail value) seen type max-field allow allowp (count 0))
    (while tail
      (when (>= count 3)
        (error "%s file-actions grammar contains too many fields" where))
      (unless (and (consp tail) (consp (cdr tail)))
        (error "%s file-actions grammar must be a proper plist" where))
      (let ((key (car tail)) (item (cadr tail)))
        (unless (memq key '(:type :max-field :allow))
          (error "%s file-actions grammar contains unknown key %S" where key))
        (when (memq key seen)
          (error "%s file-actions grammar contains duplicate key %S" where key))
        (push key seen)
        (pcase key
          (:type (setq type item))
          (:max-field (setq max-field item))
          (:allow (setq allow item allowp t))))
      (setq tail (cddr tail)
            count (1+ count)))
    (dolist (key '(:type :max-field))
      (unless (memq key seen)
        (error "%s file-actions grammar is missing key %S" where key)))
    (unless (equal type "file-actions-v1")
      (error "%s has unsupported grammar type %S" where type))
    (when (and allowp (not (stringp allow)))
      (error "%s file-actions grammar allow must be text" where))
    ;; The public grammar constructor owns the alphabet, scalar, control, and
    ;; field-bound validation shared with direct callers.
    (nl-llm-agent-grammar-file-actions max-field (and allowp allow))
    (append
     (list :type (substring-no-properties type) :max-field max-field)
     (when allowp (list :allow (substring-no-properties allow))))))

(defun nl-llm-agent-artifact--grammar-spec (value where)
  "Validate data-only grammar VALUE for WHERE and return a copy."
  ;; Preserve the legacy proper-plist/circular-list preflight before dispatch.
  (nl-llm-agent-artifact--keys
   value '(:type :length :allow :segments :max-field) where)
  (let ((type (plist-get value :type)))
    (if (equal type "file-actions-v1")
        (nl-llm-agent-artifact--file-actions-spec value where)
      ;; Preserve the exact legacy key and validation behavior for the original
      ;; done/message/template profiles.
      (nl-llm-agent-artifact--keys
       value '(:type :length :allow :segments) where)
      (unless (member type '("done" "message" "template"))
        (error "%s has unsupported grammar type %S" where type))
      (cond
       ((member type '("done" "message"))
        (let ((length (plist-get value :length))
              (allow (or (plist-get value :allow)
                         nl-llm-agent-artifact-default-allow)))
          (unless (and (integerp length) (<= 1 length) (<= length 2048))
            (error "%s grammar length must be in [1, 2048]" where))
          (unless (and (stringp allow) (not (string-empty-p allow))
                       (not (string-match-p "[\n\r]" allow)))
            (error "%s grammar allow set is invalid" where))
          (when (and (equal type "message")
                     (or (string-search "\"" allow)
                         (string-search "\\" allow)))
            (error "%s message grammar cannot allow quote or backslash" where))))
       (t
        (let ((segments (plist-get value :segments))
              (total 0))
          (unless (and (vectorp segments)
                       (<= 1 (length segments))
                       (<= (length segments) 4096))
            (error "%s template grammar requires 1..4096 segments" where))
          (dolist (segment (append segments nil))
            (cond
             ((stringp segment)
              (setq total (+ total (length segment))))
             ((listp segment)
              (nl-llm-agent-artifact--keys segment '(:slot) where)
              (let ((slot (plist-get segment :slot)))
                (unless (and (stringp slot) (not (string-empty-p slot))
                             (not (string-match-p "[\n\r]" slot)))
                  (error "%s template slot is invalid" where)))
              (setq total (1+ total)))
             (t (error "%s has invalid template segment %S" where segment))))
          (when (> total 8192)
            (error "%s template output exceeds 8192 characters" where)))))
      (copy-tree value))))

;;;###autoload
(defun nl-llm-agent-artifact-normalize-grammar (grammar &optional where)
  "Validate data-only GRAMMAR and return its detached normalized form.
WHERE customizes validation errors for an integrating service."
  (nl-llm-agent-artifact--grammar-spec
   grammar (or where "artifact grammar")))

(defun nl-llm-agent-artifact--grammar (spec)
  "Build a constrained grammar function from validated data SPEC."
  (let ((type (plist-get spec :type)))
    (cond
     ((equal type "done")
      (let ((segments (list "DONE ")))
        (dotimes (_ (plist-get spec :length))
          (setq segments
                (append segments
                        (list
                         (list :slot
                               (or (plist-get spec :allow)
                                   nl-llm-agent-artifact-default-allow))))))
        (nl-llm-agent-grammar-template segments)))
     ((equal type "message")
      (nl-llm-agent-grammar-message
       (plist-get spec :length) (plist-get spec :allow)))
     ((equal type "file-actions-v1")
      (nl-llm-agent-grammar-file-actions
       (plist-get spec :max-field) (plist-get spec :allow)))
     (t
      (nl-llm-agent-grammar-template
       (mapcar
        (lambda (segment)
          (if (stringp segment)
              segment
            (list :slot (plist-get segment :slot))))
        (append (plist-get spec :segments) nil)))))))

(defun nl-llm-agent-artifact--checkpoint-path (catalog-file relative id)
  "Resolve RELATIVE checkpoint for ID beneath CATALOG-FILE's directory."
  (unless (and (stringp relative)
               (not (string-empty-p relative))
               (not (file-name-absolute-p relative)))
    (error "artifact %s checkpoint must be a relative path" id))
  (let* ((directory
          (file-name-as-directory
           (file-name-directory (expand-file-name catalog-file))))
         (path (expand-file-name relative directory)))
    (unless (file-in-directory-p path directory)
      (error "artifact %s checkpoint escapes catalog directory" id))
    (unless (file-regular-p path)
      (error "artifact %s checkpoint does not exist: %s" id relative))
    path))

(defun nl-llm-agent-artifact--entry (value catalog-file)
  "Validate catalog model VALUE relative to CATALOG-FILE."
  (nl-llm-agent-artifact--keys
   value
   '(:id :name :checkpoint :sha256 :grammar :maxseq :score :generation)
   "artifact model")
  (let* ((id (nl-llm-agent-artifact--id
              (plist-get value :id) "artifact model"))
         (name (or (plist-get value :name) id))
         (digest (plist-get value :sha256))
         (maxseq (or (plist-get value :maxseq) 1024))
         (generation (plist-get value :generation))
         (score (plist-get value :score))
         (grammar
          (nl-llm-agent-artifact--grammar-spec
           (plist-get value :grammar) (format "artifact %s" id))))
    (unless (and (stringp name) (not (string-empty-p name)))
      (error "artifact %s name must be non-empty text" id))
    (unless (and (stringp digest)
                 (string-match-p "\\`[A-Fa-f0-9]\\{64\\}\\'" digest))
      (error "artifact %s requires a SHA-256 digest" id))
    (unless (and (integerp maxseq) (<= 1 maxseq) (<= maxseq 10000000))
      (error "artifact %s maxseq must be in [1, 10000000]" id))
    (unless (and (integerp generation) (>= generation 0))
      (error "artifact %s generation must be non-negative" id))
    (nl-llm-agent-artifact--number score (format "artifact %s score" id))
    (list :id id :name name
          :checkpoint (plist-get value :checkpoint)
          :checkpoint-path
          (nl-llm-agent-artifact--checkpoint-path
           catalog-file (plist-get value :checkpoint) id)
          :sha256 (downcase digest)
          :grammar grammar :maxseq maxseq :score score
          :generation generation)))

(defun nl-llm-agent-artifact--read (catalog-file &optional missing-ok)
  "Read and validate CATALOG-FILE, optionally accepting a missing file."
  (let ((path (expand-file-name catalog-file)))
    (if (not (file-exists-p path))
        (if missing-ok
            (list :revision 0 :models nil)
          (error "artifact catalog does not exist: %s" catalog-file))
      (unless (file-regular-p path)
        (error "artifact catalog is not a regular file: %s" catalog-file))
      (when (> (file-attribute-size (file-attributes path))
               nl-llm-agent-artifact-max-catalog-bytes)
        (error "artifact catalog exceeds %d bytes"
               nl-llm-agent-artifact-max-catalog-bytes))
      (let ((data
             (with-temp-buffer
               (insert-file-contents path)
               (json-parse-buffer
                :object-type 'plist :array-type 'array
                :null-object :json-null :false-object :json-false))))
        (nl-llm-agent-artifact--keys
         data '(:format :revision :models) "artifact catalog")
        (unless (equal (plist-get data :format)
                       nl-llm-agent-artifact-catalog-format)
          (error "unsupported artifact catalog format %S"
                 (plist-get data :format)))
        (let ((revision (plist-get data :revision))
              (models (plist-get data :models))
              (entries nil)
              (seen nil))
          (unless (and (integerp revision) (>= revision 0))
            (error "artifact catalog revision must be non-negative"))
          (unless (vectorp models)
            (error "artifact catalog requires a models array"))
          (dolist (value (append models nil))
            (let* ((entry (nl-llm-agent-artifact--entry value path))
                   (id (plist-get entry :id)))
              (when (member id seen)
                (error "duplicate artifact id %s" id))
              (setq seen (cons id seen))
              (setq entries (append entries (list entry)))))
          (list :revision revision :models entries))))))

(defun nl-llm-agent-artifact--public (entry)
  "Return model-visible descriptor for private artifact ENTRY."
  (list :id (plist-get entry :id)
        :name (plist-get entry :name)
        :capabilities '(generate local trained promoted constrained-decoding)
        :generation (plist-get entry :generation)
        :score (plist-get entry :score)
        :sha256 (plist-get entry :sha256)))

;;;###autoload
(defun nl-llm-agent-artifact-catalog (catalog-file &optional missing-ok)
  "Return public promoted model descriptors from CATALOG-FILE.
When MISSING-OK is non-nil, a missing catalog represents an empty catalog."
  (mapcar #'nl-llm-agent-artifact--public
          (plist-get
           (nl-llm-agent-artifact--read catalog-file missing-ok) :models)))

(defun nl-llm-agent-artifact--verified-model (entry)
  "Load and validate private artifact ENTRY after digest verification."
  (let* ((id (plist-get entry :id))
         (path (plist-get entry :checkpoint-path))
         (expected (plist-get entry :sha256))
         (actual (nl-llm-agent-artifact--hash-file path)))
    (unless (equal actual expected)
      (error "artifact %s digest mismatch" id))
    (nl-llm-agent-artifact--checkpoint-model
     (nl-llm-ckpt-load path) id)))

(defun nl-llm-agent-artifact--trainable-parameter (tensor)
  "Return a detached trainable autograd parameter for TENSOR."
  (photon-autograd-const
   (photon-tensor
    (copy-sequence (photon-tensor-shape tensor))
    (copy-sequence (photon-tensor-data tensor)))))

;;;###autoload
(defun nl-llm-agent-artifact-load-pav (catalog-file model-id)
  "Load promoted MODEL-ID from CATALOG-FILE as a trainable PAV model.

The artifact digest and architecture are verified before tensors become fresh
autograd leaves.  Artifact identity, generation, score, and checkpoint step are
retained as model metadata so an evolution queue can continue numbering later
generations without overwriting immutable files."
  (let* ((catalog (nl-llm-agent-artifact--read catalog-file))
         (entry
          (cl-find-if
           (lambda (item) (equal (plist-get item :id) model-id))
           (plist-get catalog :models))))
    (unless entry
      (error "unknown promoted artifact: %s" model-id))
    (let* ((checkpoint
            (nl-llm-ckpt-load (plist-get entry :checkpoint-path)))
           (inference (nl-llm-agent-artifact--verified-model entry))
           (parameter #'nl-llm-agent-artifact--trainable-parameter)
           (block-keys
            '(:ln1g :wq :bq :wk :bk :wv :bv :wo :bo
              :ln2g :wg :bg :wu :bu :wd :bd))
           (config (plist-get inference :config)))
      (unless (plist-get inference :wh)
        (error "artifact %s has no independent P5 output head" model-id))
      (unless (= (plist-get config :heads) (plist-get config :kv-heads))
        (error "artifact %s is not a P5-compatible MHA model" model-id))
      (list
       :wte (funcall parameter (plist-get inference :wte))
       :blocks
       (mapcar
        (lambda (block)
          (let ((result nil))
            (dolist (key block-keys)
              (setq result
                    (append result
                            (list key
                                  (funcall parameter
                                           (plist-get block key))))))
            result))
        (plist-get inference :blocks))
       :lnfg (funcall parameter (plist-get inference :lnfg))
       :wh (when (plist-get inference :wh)
             (funcall parameter (plist-get inference :wh)))
       :bh (funcall parameter (plist-get inference :bh))
       :dim (plist-get config :dim)
       :ff (plist-get config :ff)
       :vocab (plist-get config :vocab)
       :tokenizer (plist-get config :tokenizer)
       :heads (plist-get config :heads)
       :kv-heads (plist-get config :kv-heads)
       :kvh (plist-get config :kv-heads)
       :nblocks (plist-get config :nblocks)
       :step (or (plist-get checkpoint :step) 0)
       :artifact-id (plist-get entry :id)
       :artifact-generation (plist-get entry :generation)
       :artifact-score (plist-get entry :score)))))

(defun nl-llm-agent-artifact--hash-file (path)
  "Return lowercase SHA-256 digest for PATH."
  (with-temp-buffer
    (set-buffer-multibyte nil)
    (insert-file-contents-literally path)
    (secure-hash 'sha256 (current-buffer))))

(defun nl-llm-agent-artifact--tensor-shape-p (value shape)
  "Return non-nil when VALUE is a tensor with exact SHAPE."
  (and (nl-llm-agent-artifact--tensor-p value)
       (equal (photon-tensor-shape value) shape)))

(defun nl-llm-agent-artifact--tensor-p (value)
  "Return non-nil when VALUE has a valid photon tensor representation."
  (and (vectorp value)
       (= (length value) 2)
       (let ((shape (aref value 0))
             (data (aref value 1))
             (size 1))
         (and (listp shape)
              (not (null shape))
              (cl-every
               (lambda (dimension)
                 (when (and (integerp dimension) (> dimension 0))
                   (setq size (* size dimension))
                   t))
               shape)
              (vectorp data)
              (= (length data) size)
              (cl-every #'numberp (append data nil))))))

(defun nl-llm-agent-artifact--checkpoint-model (checkpoint id)
  "Validate CHECKPOINT for artifact ID and return an inference model."
  (let* ((config (plist-get checkpoint :config))
         (dim (plist-get config :dim))
         (heads (plist-get config :heads))
         (kvh (or (plist-get config :kv-heads)
                  (plist-get config :kvh)))
         (ff (plist-get config :ff))
         (vocab (plist-get config :vocab))
         (tokenizer
          (nl-llm-agent-tokenizer-id (plist-get config :tokenizer)))
         (nblocks (plist-get config :nblocks))
         (blocks (plist-get checkpoint :blocks)))
    (dolist (pair `((,dim . dim) (,heads . heads) (,kvh . kv-heads)
                    (,ff . ff) (,vocab . vocab) (,nblocks . nblocks)))
      (unless (and (integerp (car pair)) (> (car pair) 0))
        (error "artifact %s checkpoint has invalid %s" id (cdr pair))))
    (unless (and (= (% dim heads) 0) (<= kvh heads)
                 (= (% heads kvh) 0))
      (error "artifact %s checkpoint has incompatible attention config" id))
    (unless (= vocab (nl-llm-agent-tokenizer-vocab tokenizer))
      (error "artifact %s checkpoint vocab does not match tokenizer %s"
             id tokenizer))
    (unless (and (listp blocks) (= (length blocks) nblocks))
      (error "artifact %s checkpoint block count does not match config" id))
    (unless (nl-llm-agent-artifact--tensor-shape-p
             (plist-get checkpoint :wte) (list vocab dim))
      (error "artifact %s checkpoint has invalid embedding tensor" id))
    (unless (nl-llm-agent-artifact--tensor-shape-p
             (plist-get checkpoint :lnfg) (list dim))
      (error "artifact %s checkpoint has invalid final norm tensor" id))
    (unless (nl-llm-agent-artifact--tensor-shape-p
             (plist-get checkpoint :bh) (list vocab))
      (error "artifact %s checkpoint has invalid head bias tensor" id))
    (when (plist-get checkpoint :wh)
      (unless (nl-llm-agent-artifact--tensor-shape-p
               (plist-get checkpoint :wh) (list vocab dim))
        (error "artifact %s checkpoint has invalid output head tensor" id)))
    (let* ((head-dim (/ dim heads))
           (kvdim (* kvh head-dim))
           (shapes
            `((:ln1g ,dim) (:wq ,dim ,dim) (:bq ,dim)
              (:wk ,kvdim ,dim) (:bk ,kvdim)
              (:wv ,kvdim ,dim) (:bv ,kvdim)
              (:wo ,dim ,dim) (:bo ,dim) (:ln2g ,dim)
              (:wg ,ff ,dim) (:bg ,ff) (:wu ,ff ,dim) (:bu ,ff)
              (:wd ,dim ,ff) (:bd ,dim))))
      (dolist (block blocks)
        (dolist (item shapes)
          (unless (nl-llm-agent-artifact--tensor-shape-p
                   (plist-get block (car item)) (cdr item))
            (error "artifact %s checkpoint block has invalid %S tensor"
                   id (car item))))))
    (list :config (plist-put (copy-tree config) :tokenizer tokenizer)
          :blocks blocks :wte (plist-get checkpoint :wte)
          :wh (plist-get checkpoint :wh)
          :lnfg (plist-get checkpoint :lnfg) :bh (plist-get checkpoint :bh)
          :dim dim :heads heads :kvh kvh :tokenizer tokenizer)))

(defun nl-llm-agent-artifact--snapshot-parameter (value where)
  "Return an isolated tensor snapshot of parameter VALUE for WHERE."
  (let ((tensor (if (pav-p value) (pav-value value) value)))
    (unless (nl-llm-agent-artifact--tensor-p tensor)
      (error "%s is not a tensor parameter" where))
    (photon-tensor
     (copy-sequence (photon-tensor-shape tensor))
     (copy-sequence (photon-tensor-data tensor)))))

;;;###autoload
(defun nl-llm-agent-artifact-export-pav (model &optional step)
  "Export trainable PAV MODEL as an isolated inference checkpoint.

MODEL is the dense agent model returned by `nl-llm-agent-improve-model' or the
CPU view in `nl-llm-agent-ondevice-new'.  Its independent :wh output head is
preserved, so the exported decoder computes the same logits as the training
model.  STEP overrides MODEL's optional :step metadata."
  (let* ((dim (plist-get model :dim))
         (ff (plist-get model :ff))
         (vocab (plist-get model :vocab))
         (tokenizer
          (nl-llm-agent-tokenizer-id (plist-get model :tokenizer)))
         (heads (plist-get model :heads))
         (kvh (or (plist-get model :kv-heads)
                  (plist-get model :kvh)
                  heads))
         (nblocks (plist-get model :nblocks))
         (blocks (plist-get model :blocks))
         (block-keys
          '(:ln1g :wq :bq :wk :bk :wv :bv :wo :bo
            :ln2g :wg :bg :wu :bu :wd :bd)))
    (dolist (pair `((,dim . dim) (,ff . ff) (,vocab . vocab)
                    (,heads . heads) (,kvh . kv-heads)
                    (,nblocks . nblocks)))
      (unless (and (integerp (car pair)) (> (car pair) 0))
        (error "artifact export has invalid %s" (cdr pair))))
    (unless (and (listp blocks) (= (length blocks) nblocks))
      (error "artifact export block count does not match model metadata"))
    (unless (= vocab (nl-llm-agent-tokenizer-vocab tokenizer))
      (error "artifact export vocab does not match tokenizer %s" tokenizer))
    (let ((checkpoint
           (list
            :config
            (list :dim dim :heads heads :kv-heads kvh :ff ff
                  :vocab vocab :nblocks nblocks :tokenizer tokenizer)
            :step (if step step (or (plist-get model :step) 0))
            :wte
            (nl-llm-agent-artifact--snapshot-parameter
             (plist-get model :wte) "artifact export :wte")
            :wh
            (when (plist-get model :wh)
              (nl-llm-agent-artifact--snapshot-parameter
               (plist-get model :wh) "artifact export :wh"))
            :lnfg
            (nl-llm-agent-artifact--snapshot-parameter
             (plist-get model :lnfg) "artifact export :lnfg")
            :bh
            (nl-llm-agent-artifact--snapshot-parameter
             (plist-get model :bh) "artifact export :bh")
            :blocks
            (mapcar
             (lambda (block)
               (let ((result nil))
                 (dolist (key block-keys)
                   (setq result
                         (append
                          result
                          (list
                           key
                           (nl-llm-agent-artifact--snapshot-parameter
                            (plist-get block key)
                            (format "artifact export block %S" key))))))
                 result))
             blocks))))
      ;; Share the same structural verifier used immediately before publish.
      (nl-llm-agent-artifact--checkpoint-model
       (append (list :format nl-llm-ckpt-format) checkpoint)
       "export")
      checkpoint)))

;;;###autoload
(defun nl-llm-agent-artifact-provider
    (id catalog-file &optional name allow-missing)
  "Return a dynamic native provider ID backed by CATALOG-FILE.
Catalog changes are visible to later model discovery.  Checkpoints are loaded
only on open, after their SHA-256 digest is verified, and cached by id+digest.
ALLOW-MISSING lets a publisher create the initially empty catalog later."
  (let ((path (expand-file-name catalog-file))
        (cache nil))
    ;; Fail before registration if the current catalog is malformed.
    (nl-llm-agent-artifact--read path allow-missing)
    (nl-llm-agent-provider-new
     id
     :name (or name "Promoted NeLisp models")
     :capabilities '(generate local trained promoted constrained-decoding)
     :models
     (lambda () (nl-llm-agent-artifact-catalog path allow-missing))
     :open
     (lambda (model-id options)
       (let* ((entry
               (cl-find-if
                (lambda (item)
                  (equal (plist-get item :id) model-id))
                (plist-get (nl-llm-agent-artifact--read path) :models))))
         (unless entry
           (error "artifact provider %s: unknown model %s" id model-id))
         (let* ((digest (plist-get entry :sha256))
                (checkpoint-path (plist-get entry :checkpoint-path))
                (actual (nl-llm-agent-artifact--hash-file checkpoint-path)))
           (unless (equal actual digest)
             (error "artifact %s digest mismatch" model-id))
           (let* ((known (assoc model-id cache))
                  (cached
                   (and known
                        (equal (plist-get (cdr known) :sha256) digest)
                        (cdr known)))
                  (loaded
                   (or cached
                       (let ((value
                              (list
                               :sha256 digest
                               :model
                               (nl-llm-agent-artifact--verified-model entry)
                               :grammar
                               (nl-llm-agent-artifact--grammar
                                (plist-get entry :grammar)))))
                         (setq cache
                               (cons (cons model-id value)
                                     (cl-remove-if
                                      (lambda (cell)
                                        (equal (car cell) model-id))
                                      cache)))
                         value))))
             (nl-llm-agent-model-policy
              (plist-get loaded :model)
              (plist-get loaded :grammar)
              (or (plist-get options :maxseq)
                  (plist-get entry :maxseq)))))))
     :complete (lambda (policy messages) (funcall policy messages)))))

(defun nl-llm-agent-artifact--grammar-json (grammar)
  "Convert validated GRAMMAR plist into explicit JSON object data."
  (let ((type (plist-get grammar :type)))
    (cond
     ((equal type "file-actions-v1")
      (append
       `(("type" . ,type)
         ("max-field" . ,(plist-get grammar :max-field)))
       (when (plist-member grammar :allow)
         `(("allow" . ,(plist-get grammar :allow))))))
     ((member type '("done" "message"))
      (append
       `(("type" . ,type) ("length" . ,(plist-get grammar :length)))
       (when (plist-member grammar :allow)
         `(("allow" . ,(plist-get grammar :allow))))))
     (t
      `(("type" . "template")
        ("segments"
         . ,(apply
             #'vector
             (mapcar
              (lambda (segment)
                (if (stringp segment)
                    segment
                  `(("slot" . ,(plist-get segment :slot)))))
              (append (plist-get grammar :segments) nil)))))))))

(defun nl-llm-agent-artifact--entry-json (entry)
  "Convert normalized artifact ENTRY to JSON object data."
  `(("id" . ,(plist-get entry :id))
    ("name" . ,(plist-get entry :name))
    ("checkpoint" . ,(plist-get entry :checkpoint))
    ("sha256" . ,(plist-get entry :sha256))
    ("grammar" . ,(nl-llm-agent-artifact--grammar-json
                    (plist-get entry :grammar)))
    ("maxseq" . ,(plist-get entry :maxseq))
    ("score" . ,(plist-get entry :score))
    ("generation" . ,(plist-get entry :generation))))

(defun nl-llm-agent-artifact--write-catalog (path revision entries)
  "Atomically write PATH with REVISION and normalized ENTRIES."
  (let* ((directory (file-name-directory path))
         (temporary
          (make-temp-file (expand-file-name ".artifact-catalog-" directory))))
    (unwind-protect
        (progn
          (with-temp-file temporary
            (insert
             (json-encode
              `(("format" . ,nl-llm-agent-artifact-catalog-format)
                ("revision" . ,revision)
                ("models"
                 . ,(apply #'vector
                           (mapcar #'nl-llm-agent-artifact--entry-json
                                   entries))))))
            (terpri (current-buffer)))
          (set-file-modes temporary #o600)
          (rename-file temporary path t)
          (setq temporary nil))
      (when (and temporary (file-exists-p temporary))
        (delete-file temporary)))))

;;;###autoload
(cl-defun nl-llm-agent-artifact-publish
    (catalog-file model &key id name grammar (maxseq 1024) score generation)
  "Publish immutable checkpoint MODEL into CATALOG-FILE.
ID, data-only GRAMMAR, SCORE, and GENERATION are required.  The checkpoint is
written before an atomic catalog replacement; a failure never makes a partial
catalog entry visible."
  (let* ((path (expand-file-name catalog-file))
         (directory (file-name-directory path))
         (id (nl-llm-agent-artifact--id id "artifact publish"))
         (name (or name id))
         (grammar
          (nl-llm-agent-artifact--grammar-spec grammar "artifact publish")))
    (unless (and (stringp name) (not (string-empty-p name)))
      (error "artifact publish name must be non-empty text"))
    (unless (and (integerp maxseq) (<= 1 maxseq) (<= maxseq 10000000))
      (error "artifact publish maxseq must be in [1, 10000000]"))
    (unless (and (integerp generation) (>= generation 0))
      (error "artifact publish generation must be non-negative"))
    (nl-llm-agent-artifact--number score "artifact publish score")
    (make-directory directory t)
    (let* ((catalog (nl-llm-agent-artifact--read path t))
           (entries (plist-get catalog :models))
           (checkpoint
            (format "%s-g%06d.sexp" id generation))
           (checkpoint-path (expand-file-name checkpoint directory)))
      (when (cl-find-if
             (lambda (entry) (equal (plist-get entry :id) id)) entries)
        (error "artifact id already exists: %s" id))
      (when (file-exists-p checkpoint-path)
        (error "artifact checkpoint already exists: %s" checkpoint))
      ;; Validate before creating any persistent file.
      (nl-llm-agent-artifact--checkpoint-model
       (append (list :format nl-llm-ckpt-format) model) id)
      (let ((temporary
             (make-temp-file
              (expand-file-name ".artifact-checkpoint-" directory))))
        (unwind-protect
            (progn
              (nl-llm-ckpt-save temporary model)
              (set-file-modes temporary #o600)
              (rename-file temporary checkpoint-path nil)
              (setq temporary nil))
          (when (and temporary (file-exists-p temporary))
            (delete-file temporary))))
      (let* ((entry
              (list :id id :name name :checkpoint checkpoint
                    :checkpoint-path checkpoint-path
                    :sha256
                    (nl-llm-agent-artifact--hash-file checkpoint-path)
                    :grammar grammar :maxseq maxseq :score score
                    :generation generation))
             (updated (append entries (list entry))))
        (condition-case err
            (nl-llm-agent-artifact--write-catalog
             path (1+ (plist-get catalog :revision)) updated)
          (error
           (when (file-exists-p checkpoint-path)
             (delete-file checkpoint-path))
           (signal (car err) (cdr err))))
        (nl-llm-agent-artifact--public entry)))))

;;;###autoload
(cl-defun nl-llm-agent-artifact-publisher
    (catalog-file &key (id-prefix "champion") name grammar (maxseq 1024)
                  export)
  "Return an evolution publish hook targeting CATALOG-FILE.
Each promoted generation receives ID-PREFIX-gN.  EXPORT maps the isolated
candidate to a checkpoint model and defaults to identity."
  (nl-llm-agent-artifact--id id-prefix "artifact publisher")
  (nl-llm-agent-artifact--grammar-spec grammar "artifact publisher")
  (when (and export (not (functionp export)))
    (error "artifact publisher EXPORT must be a function"))
  (let ((export (or export (lambda (candidate) candidate))))
    (lambda (candidate promotion _state)
      (let ((generation (plist-get promotion :generation-after)))
        (nl-llm-agent-artifact-publish
         catalog-file (funcall export candidate)
         :id (format "%s-g%d" id-prefix generation)
         :name (or name (format "%s generation %d" id-prefix generation))
         :grammar grammar :maxseq maxseq
         :score (plist-get promotion :score-after)
         :generation generation)))))

(provide 'nl-llm-agent-artifact)
;;; nl-llm-agent-artifact.el ends here
