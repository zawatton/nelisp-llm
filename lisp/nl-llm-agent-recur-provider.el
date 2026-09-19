;;; nl-llm-agent-recur-provider.el --- recurrent artifact provider -*- lexical-binding: t; -*-

;;; Commentary:
;; A small provider adapter for immutable recurrent-depth artifacts.  It keeps
;; artifact loading and recurrent full-prefix decoding behind the existing
;; provider/session protocol.  This is deliberately CPU-only: it never changes
;; tensor backend dispatch and rejects a swapped GPU backend at inference time.

;;; Code:

(require 'cl-lib)
(require 'subr-x)
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-inference-runtime)
(require 'nl-llm-recur)
(require 'nl-llm-agent)
(require 'nl-llm-agent-model)
(require 'nl-llm-agent-provider)
(require 'nl-llm-agent-tokenizer)
(require 'nl-llm-agent-recur-artifact)

(declare-function nl-llm-agent-recur-artifact-load
                  "nl-llm-agent-recur-artifact" (path expected-sha256))
(declare-function nl-llm-agent-recur-supervised--model
                  "nl-llm-agent-recur-supervised" (model tokenizer))

(defconst nl-llm-agent-recur-provider--max-id 128)
(defconst nl-llm-agent-recur-provider--max-sha256 64)

(defun nl-llm-agent-recur-provider--proper-list-p (value)
  "Return non-nil when VALUE is a finite proper list."
  (let ((slow value) (fast value) (ok t))
    (while (and ok (consp fast))
      (setq fast (cdr fast))
      (when (consp fast)
        (setq fast (cdr fast) slow (cdr slow))
        (when (eq fast slow) (setq ok nil))))
    (and ok (null fast))))

(defun nl-llm-agent-recur-provider--keys (value allowed where)
  "Validate exact, non-duplicated plist VALUE against ALLOWED."
  (unless (nl-llm-agent-recur-provider--proper-list-p value)
    (error "%s must be a proper plist" where))
  (let ((tail value) seen)
    (while tail
      (let ((key (pop tail)))
        (unless (keywordp key)
          (error "%s has non-keyword key %S" where key))
        (unless tail
          (error "%s has an unpaired key %S" where key))
        (when (or (not (memq key allowed)) (memq key seen))
          (error "%s has unknown or duplicate key %S" where key))
        (push key seen)
        (pop tail))))
  value)

(defun nl-llm-agent-recur-provider--string (value where max-length)
  "Return a detached bounded string VALUE for WHERE."
  (unless (and (stringp value) (> (length value) 0)
               (<= (length value) max-length))
    (error "%s must be non-empty text of at most %d characters"
           where max-length))
  (substring-no-properties value))

(defun nl-llm-agent-recur-provider--sha256 (value where)
  "Return a detached lower-case SHA-256 string VALUE for WHERE."
  (let ((digest (nl-llm-agent-recur-provider--string
                 value where nl-llm-agent-recur-provider--max-sha256)))
    (unless (string-match-p "\\`[A-Fa-f0-9]\\{64\\}\\'" digest)
      (error "%s must be a SHA-256 digest" where))
    (downcase digest)))

(defun nl-llm-agent-recur-provider--spec (value)
  "Validate and detach one provider specification VALUE."
  (nl-llm-agent-recur-provider--keys
   value '(:id :path :sha256 :grammar :name :maxseq)
   "recurrent provider spec")
  (dolist (key '(:id :path :sha256 :grammar))
    (unless (plist-member value key)
      (error "recurrent provider spec requires %S" key)))
  (let ((id (nl-llm-agent-recur-provider--string
             (plist-get value :id) "recurrent artifact id"
             nl-llm-agent-recur-provider--max-id))
        (path (expand-file-name
               (nl-llm-agent-recur-provider--string
                (plist-get value :path) "recurrent artifact path" 4096)))
        (digest (nl-llm-agent-recur-provider--sha256
                 (plist-get value :sha256) "recurrent artifact sha256"))
        (grammar (plist-get value :grammar))
        (name (if (plist-member value :name)
                  (nl-llm-agent-recur-provider--string
                   (plist-get value :name) "recurrent artifact name" 256)
                nil))
        (maxseq (if (plist-member value :maxseq)
                    (plist-get value :maxseq)
                  128)))
    (unless (functionp grammar)
      (error "recurrent artifact %s grammar must be a function" id))
    (unless (and (integerp maxseq) (<= 1 maxseq) (<= maxseq 4096))
      (error "recurrent artifact %s maxseq must be an integer in 1..4096" id))
    (list :id id :path path :sha256 digest :grammar grammar
          :name name :maxseq maxseq)))

(defun nl-llm-agent-recur-provider--specs (specs)
  "Validate and detach provider SPECS, rejecting duplicate artifact IDs."
  (unless (nl-llm-agent-recur-provider--proper-list-p specs)
    (error "recurrent provider specs must be a proper list"))
  (let (result seen)
    (dolist (value specs (nreverse result))
      (let* ((spec (nl-llm-agent-recur-provider--spec value))
             (id (plist-get spec :id)))
        (when (member id seen)
          (error "recurrent provider has duplicate artifact id %s" id))
        (push id seen)
        (push spec result)))))

(defun nl-llm-agent-recur-provider--validate-messages (messages capacity)
  "Validate provider-neutral MESSAGES before rendering or running a model.
CAPACITY is the token bound; the exact rendered-character count prevents an
oversized concatenation before tokenizer work."
  (unless (nl-llm-agent-recur-provider--proper-list-p messages)
    (error "recurrent provider messages must be a proper list"))
  (let ((total (length "\nassistant:\n")) (first t))
    (dolist (message messages)
      (let* ((role (and (consp message) (car message)))
             (role-text (cond ((symbolp role) (symbol-name role))
                              ((stringp role) role)
                              (t nil)))
             (content (and (consp message) (cdr message))))
        (unless (and role-text (<= (length role-text) 128)
                     (stringp content) (<= (length content) (* 1024 1024)))
          (error "recurrent provider message is malformed or too large"))
        (setq total (+ total (if first 0 1)
                       (length role-text) 2 (length content))
              first nil)
        (when (> total capacity)
          (error "recurrent provider rendered prompt exceeds maxseq bound"))))
  messages))

(defun nl-llm-agent-recur-provider--public (spec)
  "Return public catalog metadata for SPEC."
  ;; The generic provider boundary adds :provider/:qualified-id.  Returning
  ;; those routing fields here would create duplicate plist keys there.
  (append (list :id (copy-sequence (plist-get spec :id))
                :name (copy-sequence
                       (or (plist-get spec :name) (plist-get spec :id)))
                :maxseq (plist-get spec :maxseq)
                :capabilities '(generate local constrained-decoding recurrent-depth))
          nil))

(defun nl-llm-agent-recur-provider--find (specs model-id)
  "Find MODEL-ID in SPECS, accepting provider protocol's string IDs."
  (let ((id (nl-llm-agent-provider--id model-id
                                        "recurrent artifact model" t)))
    (cl-find-if (lambda (spec) (equal (plist-get spec :id) id)) specs)))

(defun nl-llm-agent-recur-provider--validate-bundle (bundle)
  "Validate the common recurrent artifact BUNDLE fields and model geometry."
  (unless (and (nl-llm-agent-recur-provider--proper-list-p bundle)
               (equal (plist-get bundle :family) 'recurrent-depth)
               (plist-member bundle :model)
               (plist-member bundle :tokenizer)
               (plist-member bundle :r)
               (plist-member bundle :s0-seed)
               (plist-member bundle :step)
               (plist-member bundle :sha256))
    (error "recurrent artifact returned an invalid bundle"))
  (let ((tokenizer (nl-llm-agent-tokenizer-id (plist-get bundle :tokenizer)))
        (r (plist-get bundle :r))
        (seed (plist-get bundle :s0-seed))
        (step (plist-get bundle :step)))
    (unless (and (integerp r) (<= 1 r) (<= r 32))
      (error "recurrent artifact has invalid R"))
    (unless (and (integerp seed) (<= 0 seed) (<= seed #xffffffff))
      (error "recurrent artifact has invalid S0 seed"))
    (unless (and (integerp step) (>= step 0))
      (error "recurrent artifact has invalid checkpoint step"))
    ;; The artifact loader owns tensor geometry and hash verification; this
    ;; call additionally checks recurrent-model/tokenizer compatibility.
    (unless (fboundp 'nl-llm-agent-recur-supervised--model)
      (require 'nl-llm-agent-recur-supervised))
    (nl-llm-agent-recur-supervised--model (plist-get bundle :model) tokenizer)
    (list :model (plist-get bundle :model) :tokenizer tokenizer :r r
          :s0-seed seed :step step :family 'recurrent-depth
          :sha256 (copy-sequence (plist-get bundle :sha256)))))

(defun nl-llm-agent-recur-provider--bundle (bundle spec)
  "Validate loaded recurrent artifact BUNDLE against SPEC."
  (let ((validated (nl-llm-agent-recur-provider--validate-bundle bundle)))
    (unless (equal (downcase (plist-get validated :sha256))
                   (plist-get spec :sha256))
      (error "recurrent artifact %s bundle digest disagrees"
             (plist-get spec :id)))
    validated))

(defun nl-llm-agent-recur-provider--validate-policy-bundle (bundle)
  "Validate BUNDLE supplied directly to the public policy helper."
  (nl-llm-agent-recur-provider--validate-bundle bundle)
  bundle)

(defun nl-llm-agent-recur-provider--gpu-active-p ()
  "Return non-nil when a known photon operation is GPU-swapped."
  (if (fboundp 'nl-llm-agent-recur-supervised--gpu-active-p)
      (nl-llm-agent-recur-supervised--gpu-active-p)
    nil))

(defun nl-llm-agent-recur-provider--s0 (model sequence seed)
  "Build explicit deterministic S0 for MODEL and token SEQUENCE."
  (let ((dim (plist-get model :dim))
        (sigma (plist-get model :sigma)))
    (photon-autograd-const
     (photon-tensor
      (list (length sequence) dim)
      (nl-llm-recur-randn (* (length sequence) dim) sigma seed)))))

(defun nl-llm-agent-recur-provider--logits (state sequence)
  "Return last-row logits for STATE and full token SEQUENCE."
  (when (nl-llm-agent-recur-provider--gpu-active-p)
    (error "recurrent artifact provider refuses an active GPU"))
  ;; Prepare at the inference boundary and check dispatch again: a compiler
  ;; hook may swap a backend while the shared runtime is being refreshed.
  (nl-llm-inference-runtime-prepare)
  (when (nl-llm-agent-recur-provider--gpu-active-p)
    (error "recurrent artifact provider refuses an active GPU"))
  (let* ((bundle (plist-get state :bundle))
         (model (plist-get bundle :model))
         (result
          (let ((photon-autograd--tape nil))
            (nl-llm-recur-forward
             model sequence (plist-get bundle :r) :k 0
             :s0 (nl-llm-agent-recur-provider--s0
                  model sequence (plist-get bundle :s0-seed)))))
         (logits (plist-get result :logits))
         (tensor (pav-value logits))
         (shape (photon-tensor-shape tensor))
         (data (photon-tensor-data tensor))
         (vocab (car (photon-tensor-shape
                      (pav-value (plist-get model :wte)))))
         (start (* (1- (length sequence)) vocab)))
    (unless (and (equal shape (list (length sequence) vocab))
                 (<= (+ start vocab) (length data)))
      (error "recurrent artifact provider produced invalid logits shape"))
    (let ((out (make-vector vocab 0.0)) (index 0))
      (while (< index vocab)
        (let ((value (aref data (+ start index))))
          (unless (and (numberp value) (= value value)
                       (< (abs (float value)) 1.0e308))
            (error "recurrent artifact provider produced non-finite logits"))
          (aset out index value))
        (setq index (1+ index)))
      out)))

(defun nl-llm-agent-recur-provider-policy (bundle grammar &optional maxseq)
  "Return a full-prefix recurrent policy for BUNDLE under GRAMMAR.
Each generated token recomputes the complete prefix with a deterministic S0;
the policy is CPU-only and does not mutate BUNDLE."
  (unless (functionp grammar)
    (error "invalid recurrent provider policy arguments"))
  (nl-llm-agent-recur-provider--validate-policy-bundle bundle)
  (let* ((tokenizer (plist-get bundle :tokenizer))
         (capacity (or maxseq 128)))
    (unless (and (integerp capacity) (<= 1 capacity) (<= capacity 4096))
      (error "recurrent provider maxseq must be an integer in 1..4096"))
    (lambda (messages)
      (nl-llm-agent-recur-provider--validate-messages messages capacity)
      (let* ((prompt (nl-llm-agent--render messages))
             (prefix (nl-llm-agent-tokenizer-encode prompt tokenizer)))
        (when (> (length prefix) capacity)
          (error "recurrent provider prompt exceeds maxseq"))
        (cl-labels
            ((forward ()
               (when (> (length prefix) capacity)
                 (error "recurrent provider sequence exceeds maxseq"))
               (nl-llm-agent-recur-provider--logits
                (list :bundle bundle) prefix))
             (step (token)
               (when (>= (length prefix) capacity)
                 (error "recurrent provider sequence exceeds maxseq"))
               (setq prefix (append prefix (list token)))
               (forward)))
          (nl-llm-agent-constrained-generate
           (forward) #'step grammar tokenizer))))))

;;;###autoload
(defun nl-llm-agent-recur-policy (bundle grammar &optional maxseq)
  "Public alias for `nl-llm-agent-recur-provider-policy'."
  (nl-llm-agent-recur-provider-policy bundle grammar maxseq))

;;;###autoload
(defun nl-llm-agent-recur-provider (id specs &optional name)
  "Return provider ID backed by immutable recurrent artifact SPECS.
Each spec contains :id, :path, :sha256, and trusted function :grammar, with
optional :name and :maxseq.  Artifact paths and grammar closures are private;
the public provider catalog exposes only descriptive metadata."
  (let* ((provider-id (nl-llm-agent-provider--id id
                                                 "recurrent provider"))
         (normalized (nl-llm-agent-recur-provider--specs specs)))
    (nl-llm-agent-provider-new
     provider-id :name (or name provider-id)
     :capabilities '(generate local constrained-decoding recurrent-depth)
     :models (lambda ()
               (mapcar
                (lambda (spec)
                  (nl-llm-agent-recur-provider--public spec))
                normalized))
     :open
     (lambda (model-id options)
       (unless (or (null options)
                   (and (nl-llm-agent-recur-provider--proper-list-p options)
                        (= (length options) 2)
                        (eq (car options) :maxseq)))
         (error "recurrent provider options must be nil or :maxseq"))
       (let* ((spec (nl-llm-agent-recur-provider--find normalized model-id))
              (capacity (if options (cadr options) (plist-get spec :maxseq))))
         (unless spec
           (error "recurrent provider %s: unknown model %s"
                  provider-id model-id))
         (unless (and (integerp capacity) (<= 1 capacity) (<= capacity 4096))
           (error "recurrent provider maxseq must be an integer in 1..4096"))
         (let ((bundle
                (nl-llm-agent-recur-provider--bundle
                 (nl-llm-agent-recur-artifact-load
                  (plist-get spec :path) (plist-get spec :sha256))
                 spec)))
           (list :bundle bundle :maxseq capacity :artifact-id
                 (copy-sequence (plist-get spec :id))))))
     :complete
     (lambda (state messages)
       (funcall
        (nl-llm-agent-recur-provider-policy
         (plist-get state :bundle)
         (plist-get
          (cl-find-if
           (lambda (spec)
             (equal (plist-get spec :id)
                    (plist-get state :artifact-id)))
           normalized)
          :grammar)
         (plist-get state :maxseq))
        messages)))))

(provide 'nl-llm-agent-recur-provider)
;;; nl-llm-agent-recur-provider.el ends here
