;;; weights-header-test.el --- the exported weight table is Elisp-readable  -*- lexical-binding: t; -*-
;; Run: emacs -Q --batch -l test/weights-header-test.el
;;
;; Doc 08 Phase 2b.  The table's header is one Lisp sexp precisely so the Phase
;; 2c loader needs no parser, and this pins that claim: `read' must accept it,
;; the config must round-trip as the numbers config.json holds, and every
;; tensor's declared extent must fall inside the file.
;;
;; The numeric content of the table is checked by
;; tools/qwen-weights-verify.py, which reads the payload against the donor and
;; is calibrated by corrupting a byte and a scale.  This suite is about the
;; format being consumable from Elisp, which no Python check can establish.
;;
;; The table is donor-derived and gitignored, so this skips rather than fails
;; when it is absent.

(load (expand-file-name "../lisp/nl-llm-stack-paths.el"
                        (file-name-directory (or load-file-name buffer-file-name
                                                 default-directory))) nil t)

(defvar wh--fail 0)
(defvar wh--table (expand-file-name "build/donor/qwen3-0.6b/weights.bin"))
(defvar wh--config (expand-file-name "build/donor/qwen3-0.6b/config.json"))

(defun wh--ck (name ok &optional extra)
  (princ (format "%-46s %s  %s\n" name
                 (if ok "PASS" (progn (setq wh--fail (1+ wh--fail)) "FAIL"))
                 (or extra ""))))

(defconst wh--magic "nl-llm-wts-v1")

(defun wh--u32 (s i)
  (+ (aref s i) (ash (aref s (+ i 1)) 8)
     (ash (aref s (+ i 2)) 16) (ash (aref s (+ i 3)) 24)))

(defun wh--read-head (path bytes)
  "Return the first BYTES of PATH as a unibyte string."
  (with-temp-buffer
    (set-buffer-multibyte nil)
    (let ((coding-system-for-read 'binary))
      (insert-file-contents-literally path nil 0 bytes))
    (buffer-substring-no-properties (point-min) (point-max))))

(if (not (file-readable-p wh--table))
    (princ (format "SKIP: weight table missing\n  %s\n\
  regenerate with:  make qwen-weights-table\n" wh--table))

  (let* ((probe (wh--read-head wh--table 32))
         (hlen (wh--u32 probe 16))
         (raw (wh--read-head wh--table (+ 20 hlen)))
         (total (file-attribute-size (file-attributes wh--table)))
         (payload-at (+ 20 hlen))
         hdr tensors)

    (wh--ck "magic" (string-prefix-p wh--magic raw)
            (substring raw 0 (length wh--magic)))

    ;; The whole point of a sexp header: no parser, just `read'.
    (let ((text (decode-coding-string (substring raw 20 (+ 20 hlen))
                                      'utf-8-unix)))
      (condition-case err
          (setq hdr (car (read-from-string text)))
        (error (wh--ck "header is readable by `read'" nil (format "%S" err)))))
    (when hdr
      (wh--ck "header is readable by `read'" t (format "%d bytes" hlen))
      (setq tensors (plist-get hdr :tensors))

      ;; Config must survive the trip as numbers, not strings.
      (let ((want (with-temp-buffer
                    (let ((coding-system-for-read 'utf-8-unix))
                      (insert-file-contents wh--config))
                    (goto-char (point-min))
                    (json-parse-buffer :object-type 'alist))))
        (dolist (pair '((:dim . hidden_size) (:heads . num_attention_heads)
                        (:kv-heads . num_key_value_heads)
                        (:head-dim . head_dim) (:layers . num_hidden_layers)
                        (:ff . intermediate_size) (:vocab . vocab_size)))
          (let ((got (plist-get hdr (car pair)))
                (exp (alist-get (cdr pair) want)))
            (wh--ck (format "config %s" (car pair))
                    (and (integerp got) (equal got exp))
                    (format "%S (config %S)" got exp))))
        (wh--ck "config :rope-base"
                (and (floatp (plist-get hdr :rope-base))
                     (= (plist-get hdr :rope-base)
                        (float (alist-get 'rope_theta want))))
                (format "%S" (plist-get hdr :rope-base))))

      ;; This is the shape Phase 2a made representable; if the header lost it
      ;; the loader would silently fall back to (/ dim heads).
      (wh--ck "head-dim is decoupled from dim/heads"
              (and (plist-get hdr :head-dim)
                   (/= (plist-get hdr :head-dim)
                       (/ (plist-get hdr :dim) (plist-get hdr :heads))))
              (format "head-dim %S vs dim/heads %S"
                      (plist-get hdr :head-dim)
                      (/ (plist-get hdr :dim) (plist-get hdr :heads))))

      (wh--ck "tensor count" (> (length tensors) 0)
              (format "%d" (length tensors)))

      ;; Every declared extent must be inside the file, and int8 tensors must
      ;; carry one scale per output row.  A header that overruns its own
      ;; payload is the failure a loader would hit as garbage, far from here.
      (let ((bad nil) (roles nil) (int8 0) (f32 0))
        (dolist (tn tensors)
          (let* ((kind (plist-get tn :kind))
                 (shape (plist-get tn :shape))
                 (off (plist-get tn :offset))
                 (nb (plist-get tn :nbytes))
                 (so (plist-get tn :scale-offset))
                 (end (+ payload-at off nb)))
            (push (plist-get tn :role) roles)
            (unless (and (integerp off) (integerp nb) (> nb 0) (<= end total))
              (push (format "%s extent %d+%d past %d"
                            (plist-get tn :name) off nb total)
                    bad))
            (cond
             ((equal kind "int8x4")
              (setq int8 (1+ int8))
              (let ((want-scales (* 4 (car shape))))
                (unless (and (integerp so) (>= so 0)
                             (<= (+ payload-at so want-scales) total))
                  (push (format "%s scales %S + %d past %d"
                                (plist-get tn :name) so want-scales total)
                        bad)))
              ;; Four int8 lanes per word, rows padded up to a whole word.
              (let ((want-bytes (* (car shape) (plist-get tn :words) 4)))
                (unless (= nb want-bytes)
                  (push (format "%s nbytes %d != %d"
                                (plist-get tn :name) nb want-bytes)
                        bad))))
             ((equal kind "f32")
              (setq f32 (1+ f32))
              (let ((want-bytes (* 4 (apply #'* shape))))
                (unless (= nb want-bytes)
                  (push (format "%s f32 nbytes %d != %d"
                                (plist-get tn :name) nb want-bytes)
                        bad))))
             (t (push (format "%s unknown kind %S" (plist-get tn :name) kind)
                      bad)))))
        (wh--ck "every tensor extent is inside the file"
                (null bad)
                (if bad (format "%d problems: %s" (length bad)
                                (car (nreverse bad)))
                  (format "%d int8x4, %d f32" int8 f32)))

        ;; The roles are what the model plist is built from, so a missing one
        ;; is a model that cannot be assembled.
        (let* ((have (delete-dups (nreverse roles)))
               (need '(:wq :wk :wv :wo :wg :wu :wd :ln1g :ln2g
                       :q-norm :k-norm :wte :lnf))
               (absent (seq-remove (lambda (r) (memq r have)) need)))
          (wh--ck "every role the model needs is present"
                  (null absent)
                  (if absent (format "missing %S" absent)
                    (format "%d roles" (length have)))))

        ;; Tied donors must not ship a separate head role; the exporter proves
        ;; the duplicate is one before dropping it.
        (wh--ck "tied head means no :head role"
                (if (plist-get hdr :tied-head)
                    (not (memq :head (delete-dups roles)))
                  (memq :head (delete-dups roles)))
                (format "tied-head %S" (plist-get hdr :tied-head)))))

    (princ (format "\n%s: %d failure(s)\n"
                   (if (zerop wh--fail) "weights-header OK" "weights-header")
                   wh--fail))
    (when (> wh--fail 0) (kill-emacs 1))))
