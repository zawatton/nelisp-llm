;;; nl-llm-weights.el --- read an imported int8 weight table  -*- lexical-binding: t; -*-

;; Doc 08 (docs/design/08-weight-import.org) Phase 2c: the pure-Elisp side of
;; weight import.  Reads the `nl-llm-wts-v1' table written by
;; tools/qwen-weights-export.py -- a 16-byte magic, a u32 header length, one
;; Lisp sexp of metadata, then a payload of per-output-row int8 (four lanes per
;; little-endian u32) and f32 scale rows.
;;
;; The design constraint is what this file must NOT do.  Qwen3-0.6B is 596M
;; parameters; a `photon-tensor' holds boxed floats at roughly 24 bytes each, so
;; materializing the model that way needs about 14 GB.  Nothing here builds a
;; full-precision vector for a large tensor.  Instead a tensor is addressed by
;; byte range in the file and handed out either as raw bytes (for upload to a
;; GPU buffer verbatim -- the packing is already what the DP4A kernel reads via
;; `bitcast-u') or one dequantized row at a time.
;;
;; Row-at-a-time is not only a memory trick: it is what makes the import
;; checkable.  A row can be compared against the exporter's own dequantization,
;; which separates "Elisp unpacked the wrong lane or the wrong scale" from
;; "int8 is lossy", two failures that otherwise both read as slightly wrong
;; numbers.  test/weights-load-test.el does exactly that.
;;
;; Not here yet: bulk upload to a resident GPU buffer.  The server's
;; `nelisp-gpu-server-upload-u32' takes a vector of Elisp integers and lists it
;; before encoding, which for this payload is ~149M words and several GB of
;; heap; a bytes-in primitive belongs on the nelisp-gpu side.  See Phase 2c in
;; the design doc.

;;; Code:

(require 'cl-lib)
(require 'nl-llm-compat)

;; photon-tensor is needed only by `nl-llm-weights-f32-tensor', which requires
;; it at call time; declaring it keeps this file loadable without the tensor
;; library for the byte-range and row paths, which are the ones that matter for
;; a 598 MB table.
(declare-function photon-tensor "photon-tensor" (shape data))

(defconst nl-llm-weights-magic "nl-llm-wts-v1"
  "Magic prefix of an exported weight table.")

(cl-defstruct (nl-llm-weights (:constructor nl-llm-weights--make))
  path         ; the table file
  header       ; the whole header plist
  payload-at   ; byte offset where tensor payload begins
  size         ; total file size, for extent checks
  index)       ; hash (ROLE . LAYER) -> tensor plist

;;; --- primitive decoding --------------------------------------------------

(defsubst nl-llm-weights--u32 (s i)
  "Read a little-endian unsigned 32-bit integer from unibyte S at index I."
  (+ (aref s i) (ash (aref s (+ i 1)) 8)
     (ash (aref s (+ i 2)) 16) (ash (aref s (+ i 3)) 24)))

(defsubst nl-llm-weights--i8 (b)
  "Interpret byte B as a signed 8-bit integer."
  (if (> b 127) (- b 256) b))

(defun nl-llm-weights--f32 (s i)
  "Decode the IEEE-754 binary32 at index I of unibyte S.
Done arithmetically because Emacs has no float-from-bytes primitive; `ldexp'
supplies the power of two exactly, so a normal value round-trips."
  (let* ((bits (nl-llm-weights--u32 s i))
         (neg (/= 0 (logand bits #x80000000)))
         (expo (logand (ash bits -23) #xff))
         (mant (logand bits #x7fffff))
         (mag (cond
               ((= expo 0)
                (if (= mant 0) 0.0 (ldexp (/ mant 8388608.0) -126)))
               ((= expo 255)
                ;; Infinities and NaNs have no place in a weight table; a donor
                ;; that produced one is a conversion to reject, not to average.
                (error "nl-llm-weights: non-finite float in table at byte %d" i))
               (t (ldexp (+ 1.0 (/ mant 8388608.0)) (- expo 127))))))
    (if neg (- mag) mag)))

(defun nl-llm-weights--slice (path beg end)
  "Return bytes [BEG, END) of PATH as a unibyte string."
  (with-temp-buffer
    (set-buffer-multibyte nil)
    (let ((coding-system-for-read 'binary))
      (insert-file-contents-literally path nil beg end))
    (buffer-substring-no-properties (point-min) (point-max))))

;;; --- opening -------------------------------------------------------------

;;;###autoload
(defun nl-llm-weights-open (path)
  "Open the weight table at PATH and return an `nl-llm-weights' handle.
Reads the header only; tensor payload stays on disk until asked for."
  (unless (file-readable-p path)
    (error "nl-llm-weights-open: cannot read %s" path))
  (let* ((probe (nl-llm-weights--slice path 0 20))
         (size (file-attribute-size (file-attributes path))))
    (unless (string-prefix-p nl-llm-weights-magic probe)
      (error "nl-llm-weights-open: %s is not an %s table"
             path nl-llm-weights-magic))
    (let* ((hlen (nl-llm-weights--u32 probe 16))
           (payload-at (+ 20 hlen))
           (text (decode-coding-string
                  (nl-llm-weights--slice path 20 payload-at) 'utf-8-unix))
           (header (car (read-from-string text)))
           (index (make-hash-table :test 'equal)))
      (dolist (tn (plist-get header :tensors))
        (let ((key (cons (plist-get tn :role) (plist-get tn :layer))))
          (when (gethash key index)
            (error "nl-llm-weights-open: duplicate tensor %S" key))
          (puthash key tn index)))
      (nl-llm-weights--make :path path :header header :payload-at payload-at
                            :size size :index index))))

;;;###autoload
(defun nl-llm-weights-config (wts)
  "Return the model configuration of WTS as a plist.
Shaped so it can be spliced into the model plist `nl-llm-model-forward' reads:
:dim :heads :kv-heads :head-dim :rope-base, plus :layers :ff :vocab
:rms-eps :tied-head for the caller to assemble with."
  (let ((h (nl-llm-weights-header wts)))
    (list :dim (plist-get h :dim)
          :heads (plist-get h :heads)
          :kv-heads (plist-get h :kv-heads)
          :head-dim (plist-get h :head-dim)
          :rope-base (plist-get h :rope-base)
          :layers (plist-get h :layers)
          :ff (plist-get h :ff)
          :vocab (plist-get h :vocab)
          :rms-eps (plist-get h :rms-eps)
          :tied-head (plist-get h :tied-head))))

;;;###autoload
(defun nl-llm-weights-tensor (wts role &optional layer)
  "Return WTS's tensor plist for ROLE at LAYER (nil or -1 for a top-level one)."
  (or (gethash (cons role (or layer -1)) (nl-llm-weights-index wts))
      (error "nl-llm-weights: no tensor for role %S layer %S" role layer)))

(defun nl-llm-weights--extent (wts tn)
  "Return (BEG . END) of TN's packed payload within WTS's file."
  (let* ((beg (+ (nl-llm-weights-payload-at wts) (plist-get tn :offset)))
         (end (+ beg (plist-get tn :nbytes))))
    (unless (<= end (nl-llm-weights-size wts))
      (error "nl-llm-weights: %s extends past the file (%d > %d)"
             (plist-get tn :name) end (nl-llm-weights-size wts)))
    (cons beg end)))

;;;###autoload
(defun nl-llm-weights-bytes (wts tn)
  "Return TN's payload from WTS verbatim, as a unibyte string.
For an int8x4 tensor these bytes are already the u32 words a DP4A kernel reads
with `bitcast-u', so an uploader can hand them over without touching a float."
  (let ((ext (nl-llm-weights--extent wts tn)))
    (nl-llm-weights--slice (nl-llm-weights-path wts) (car ext) (cdr ext))))

;;;###autoload
(defun nl-llm-weights-scales (wts tn)
  "Return TN's per-output-row scales from WTS as a float vector.
Signals for an f32 tensor, which has no scales."
  (unless (equal (plist-get tn :kind) "int8x4")
    (error "nl-llm-weights: %s is %s, not int8x4"
           (plist-get tn :name) (plist-get tn :kind)))
  (let* ((rows (car (plist-get tn :shape)))
         (beg (+ (nl-llm-weights-payload-at wts) (plist-get tn :scale-offset)))
         (raw (nl-llm-weights--slice (nl-llm-weights-path wts)
                                     beg (+ beg (* 4 rows))))
         (out (make-vector rows 0.0)))
    (dotimes (i rows)
      (aset out i (nl-llm-weights--f32 raw (* 4 i))))
    out))

;;;###autoload
(defun nl-llm-weights-lanes (wts tn row)
  "Return TN's ROW as its raw signed int8 lanes, a vector of COLS integers.
The padding lanes that round a row up to a whole word are not returned."
  (unless (equal (plist-get tn :kind) "int8x4")
    (error "nl-llm-weights: %s is %s, not int8x4"
           (plist-get tn :name) (plist-get tn :kind)))
  (let* ((shape (plist-get tn :shape))
         (rows (car shape)) (cols (nth 1 shape))
         (words (plist-get tn :words)))
    (unless (and (integerp row) (>= row 0) (< row rows))
      (error "nl-llm-weights: row %S outside 0..%d" row (1- rows)))
    (let* ((stride (* 4 words))
           (beg (+ (car (nl-llm-weights--extent wts tn)) (* row stride)))
           (raw (nl-llm-weights--slice (nl-llm-weights-path wts)
                                       beg (+ beg stride)))
           (out (make-vector cols 0)))
      (dotimes (i cols) (aset out i (nl-llm-weights--i8 (aref raw i))))
      out)))

;;;###autoload
(defun nl-llm-weights-row (wts tn row &optional scales)
  "Return TN's ROW from WTS dequantized, as a float vector of COLS.
SCALES, when given, is the vector from `nl-llm-weights-scales' -- pass it when
walking many rows so the scale block is read once.  For an f32 tensor the row is
returned as stored."
  (if (equal (plist-get tn :kind) "f32")
      (let* ((shape (plist-get tn :shape))
             (cols (or (nth 1 shape) (car shape)))
             (beg (+ (car (nl-llm-weights--extent wts tn))
                     (* 4 cols (if (nth 1 shape) row 0))))
             (raw (nl-llm-weights--slice (nl-llm-weights-path wts)
                                         beg (+ beg (* 4 cols))))
             (out (make-vector cols 0.0)))
        (dotimes (i cols) (aset out i (nl-llm-weights--f32 raw (* 4 i))))
        out)
    (let* ((lanes (nl-llm-weights-lanes wts tn row))
           (scale (aref (or scales (nl-llm-weights-scales wts tn)) row))
           (n (length lanes))
           (out (make-vector n 0.0)))
      (dotimes (i n) (aset out i (* (aref lanes i) scale)))
      out)))

;;;###autoload
(defun nl-llm-weights-f32-tensor (wts tn)
  "Return an f32 TN from WTS as a `photon-tensor'.
Only for the small unquantized tensors -- RMSNorm gains and the Qwen3
q_norm/k_norm vectors.  An int8x4 tensor is refused: dequantizing one into boxed
floats is the 14 GB mistake this file exists to avoid."
  (unless (equal (plist-get tn :kind) "f32")
    (error "nl-llm-weights-f32-tensor: %s is int8x4; use `nl-llm-weights-row' \
or `nl-llm-weights-bytes'" (plist-get tn :name)))
  (require 'photon-tensor)
  (let* ((shape (plist-get tn :shape))
         (n (apply #'* shape))
         (ext (nl-llm-weights--extent wts tn))
         (raw (nl-llm-weights--slice (nl-llm-weights-path wts)
                                     (car ext) (cdr ext)))
         (v (make-vector n 0.0)))
    (dotimes (i n) (aset v i (nl-llm-weights--f32 raw (* 4 i))))
    (photon-tensor (copy-sequence shape) v)))

(provide 'nl-llm-weights)
;;; nl-llm-weights.el ends here
