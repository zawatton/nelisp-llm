;;; nl-llm-agent-tokenizer.el --- bounded native agent tokenizers -*- lexical-binding: t; -*-

;; Native agent models currently use either the original printable-ASCII
;; character vocabulary or the 256 unmerged base tokens of photon-bpe.

;;; Code:

(require 'photon-bpe)

(defconst nl-llm-agent-tokenizer-ascii "ascii-char-v1"
  "Identifier for the legacy printable-ASCII character tokenizer.")

(defconst nl-llm-agent-tokenizer-utf8 "utf8-byte-v1"
  "Identifier for the strict UTF-8 byte tokenizer.")

(defun nl-llm-agent-tokenizer-id (&optional identifier)
  "Return the canonical tokenizer string for IDENTIFIER.
Nil retains the legacy `ascii-char-v1' tokenizer."
  (let ((canonical
         (cond
          ((or (null identifier)
               (equal identifier nl-llm-agent-tokenizer-ascii))
           nl-llm-agent-tokenizer-ascii)
          ((equal identifier nl-llm-agent-tokenizer-utf8)
           nl-llm-agent-tokenizer-utf8)
          (t (error "Unsupported native agent tokenizer %S" identifier)))))
    ;; Never expose a mutable identifier string owned by the caller or module.
    (apply #'string (string-to-list canonical))))

(defun nl-llm-agent-tokenizer-vocab (&optional identifier)
  "Return the vocabulary size for tokenizer IDENTIFIER."
  (if (equal (nl-llm-agent-tokenizer-id identifier)
             nl-llm-agent-tokenizer-utf8)
      256
    96))

(defun nl-llm-agent-tokenizer--validate-text (text)
  "Reject non-string TEXT and characters that are not Unicode scalars."
  (unless (stringp text)
    (error "Tokenizer text must be a string, got %S" text))
  (let ((multibyte (multibyte-string-p text))
        (chars (string-to-list text)))
    (dolist (char chars)
      (unless (and (integerp char)
                   (>= char 0)
                   (<= char #x10ffff)
                   (not (and (>= char #xd800) (<= char #xdfff)))
                   (or multibyte (< char 128)))
        (error "Tokenizer text contains a non-Unicode-scalar character: %S"
               char))))
  text)

(defun nl-llm-agent-tokenizer--ascii-id (char)
  "Return legacy ASCII token id for CHAR, rejecting unsupported input."
  (cond
   ((= char ?\n) 95)
   ((and (>= char 32) (<= char 126)) (- char 32))
   (t (error "ascii-char-v1 cannot encode character U+%04X" char))))

;;;###autoload
(defun nl-llm-agent-tokenizer-encode (text &optional identifier)
  "Encode TEXT with IDENTIFIER and return a fresh list of token ids."
  (nl-llm-agent-tokenizer--validate-text text)
  (if (equal (nl-llm-agent-tokenizer-id identifier)
             nl-llm-agent-tokenizer-utf8)
      ;; With no merges, photon-bpe's base ids are exactly its UTF-8 bytes.
      (copy-sequence (photon-bpe--bytes text))
    (mapcar #'nl-llm-agent-tokenizer--ascii-id (string-to-list text))))

(defun nl-llm-agent-tokenizer--ids (ids limit)
  "Return a fresh proper copy of IDS after validating values below LIMIT."
  (let ((slow ids)
        (fast ids))
    ;; Reject cycles and dotted tails before walking the list to copy it.
    (while (consp fast)
      (setq fast (cdr fast))
      (when (eq fast slow)
        (error "Tokenizer ids must not be circular"))
      (when (consp fast)
        (setq fast (cdr fast)
              slow (cdr slow))
        (when (eq fast slow)
          (error "Tokenizer ids must not be circular"))))
    (unless (null fast)
      (error "Tokenizer ids must be a proper list")))
  (let ((tail ids)
        (result nil))
    (while (consp tail)
      (let ((id (car tail)))
        (unless (and (integerp id) (>= id 0) (< id limit))
          (error "Tokenizer id must be an integer in [0,%d), got %S"
                 limit id))
        (push id result))
      (setq tail (cdr tail)))
    (nreverse result)))

(defun nl-llm-agent-tokenizer--continuation-p (byte)
  "Return non-nil when BYTE is a UTF-8 continuation byte."
  (and (>= byte #x80) (<= byte #xbf)))

(defun nl-llm-agent-tokenizer--utf8-codepoints (bytes)
  "Strictly decode validated byte-list BYTES to Unicode code points."
  (let ((tail bytes)
        (result nil))
    (while tail
      (let ((b0 (pop tail)))
        (cond
         ((<= b0 #x7f)
          (push b0 result))
         ((and (>= b0 #xc2) (<= b0 #xdf))
          (unless (and tail
                       (nl-llm-agent-tokenizer--continuation-p (car tail)))
            (error "Malformed or truncated UTF-8 sequence"))
          (push (+ (ash (logand b0 #x1f) 6)
                   (logand (pop tail) #x3f))
                result))
         ((and (>= b0 #xe0) (<= b0 #xef))
          (unless (and (consp tail) (consp (cdr tail)))
            (error "Truncated UTF-8 sequence"))
          (let ((b1 (pop tail))
                (b2 (pop tail)))
            (unless (and (nl-llm-agent-tokenizer--continuation-p b1)
                         (nl-llm-agent-tokenizer--continuation-p b2)
                         (if (= b0 #xe0) (>= b1 #xa0) t)
                         (if (= b0 #xed) (<= b1 #x9f) t))
              (error "Malformed, overlong, or surrogate UTF-8 sequence"))
            (push (+ (ash (logand b0 #x0f) 12)
                     (ash (logand b1 #x3f) 6)
                     (logand b2 #x3f))
                  result)))
         ((and (>= b0 #xf0) (<= b0 #xf4))
          (unless (and (consp tail)
                       (consp (cdr tail))
                       (consp (cddr tail)))
            (error "Truncated UTF-8 sequence"))
          (let ((b1 (pop tail))
                (b2 (pop tail))
                (b3 (pop tail)))
            (unless (and (nl-llm-agent-tokenizer--continuation-p b1)
                         (nl-llm-agent-tokenizer--continuation-p b2)
                         (nl-llm-agent-tokenizer--continuation-p b3)
                         (if (= b0 #xf0) (>= b1 #x90) t)
                         (if (= b0 #xf4) (<= b1 #x8f) t))
              (error "Malformed, overlong, or out-of-range UTF-8 sequence"))
            (push (+ (ash (logand b0 #x07) 18)
                     (ash (logand b1 #x3f) 12)
                     (ash (logand b2 #x3f) 6)
                     (logand b3 #x3f))
                  result)))
         (t
          (error "Invalid UTF-8 leading byte 0x%02X" b0)))))
    (nreverse result)))

;;;###autoload
(defun nl-llm-agent-tokenizer-decode (ids &optional identifier)
  "Strictly decode token IDS with tokenizer IDENTIFIER to text."
  (let* ((id (nl-llm-agent-tokenizer-id identifier))
         (values (nl-llm-agent-tokenizer--ids
                  ids (nl-llm-agent-tokenizer-vocab id))))
    (if (equal id nl-llm-agent-tokenizer-utf8)
        (apply #'string
               (nl-llm-agent-tokenizer--utf8-codepoints values))
      (apply #'string
             (mapcar (lambda (value)
                       (if (= value 95) ?\n (+ value 32)))
                     values)))))

(provide 'nl-llm-agent-tokenizer)
;;; nl-llm-agent-tokenizer.el ends here
