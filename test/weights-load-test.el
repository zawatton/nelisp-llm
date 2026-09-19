;;; weights-load-test.el --- the Elisp reader unpacks what the exporter packed  -*- lexical-binding: t; -*-
;; Run: emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/weights-load-test.el
;;
;; Doc 08 Phase 2c.  lisp/nl-llm-weights.el has three things it can get wrong
;; while still returning numbers: which lane of a packed word it reads, which
;; per-output-row scale it pairs with a row, and how it turns four bytes into a
;; float.  Each is checked separately against tools/qwen-weights-rows.py, which
;; samples the same rows straight out of the table with numpy:
;;
;;   :lanes   integers, so a mis-indexed unpack cannot hide behind rounding
;;   :scale   the row's own scale, compared exactly
;;   :values  the float64 product, so no tolerance is needed and therefore no
;;            tolerance is available to hide an indexing bug in
;;
;; The fixture's products are float64 deliberately: int8 * f32_scale can need
;; more than 24 mantissa bits, so a float32 reference would disagree with
;; Emacs's arithmetic and force a tolerance.  The GPU multiplies in f32; that is
;; Phase 2d's concern, not the reader's.
;;
;; Table and fixture are donor-derived and gitignored, so this skips rather than
;; fails when they are absent.

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'nl-llm-weights)

(defvar wl--fail 0)
(defvar wl--table (expand-file-name "build/donor/qwen3-0.6b/weights.bin"))
(defvar wl--fixture (expand-file-name "test/fixtures/qwen-weights-rows.eld"))

(defun wl--ck (name ok &optional extra)
  (princ (format "%-50s %s  %s\n" name
                 (if ok "PASS" (progn (setq wl--fail (1+ wl--fail)) "FAIL"))
                 (or extra ""))))

(if (not (and (file-readable-p wl--table) (file-readable-p wl--fixture)))
    (princ (format "SKIP: weight table or row fixture missing\n  %s\n  %s\n\
  regenerate with:  make qwen-weights-table && make qwen-weights-rows\n"
                   wl--table wl--fixture))

  (let* ((t0 (float-time))
         (wts (nl-llm-weights-open wl--table))
         (open-secs (- (float-time) t0))
         (cfg (nl-llm-weights-config wts))
         (samples (with-temp-buffer
                    (let ((coding-system-for-read 'utf-8-unix))
                      (insert-file-contents wl--fixture))
                    (goto-char (point-min))
                    (read (current-buffer)))))

    (wl--ck "table opens" (nl-llm-weights-p wts)
            (format "%d tensors, %.2fs (header only)"
                    (length (plist-get (nl-llm-weights-header wts) :tensors))
                    open-secs))

    ;; The config must arrive as the model plist wants it, head-dim included --
    ;; that is the value Phase 2a made meaningful.
    (wl--ck "config carries the decoupled head width"
            (and (= (plist-get cfg :dim) 1024)
                 (= (plist-get cfg :heads) 16)
                 (= (plist-get cfg :kv-heads) 8)
                 (= (plist-get cfg :head-dim) 128)
                 (/= (plist-get cfg :head-dim)
                     (/ (plist-get cfg :dim) (plist-get cfg :heads))))
            (format "dim %S heads %S kv %S head-dim %S"
                    (plist-get cfg :dim) (plist-get cfg :heads)
                    (plist-get cfg :kv-heads) (plist-get cfg :head-dim)))

    (wl--ck "config carries rope-base and tied-head"
            (and (floatp (plist-get cfg :rope-base))
                 (= (plist-get cfg :rope-base) 1000000.0)
                 (eq (plist-get cfg :tied-head) t))
            (format "%S / tied %S" (plist-get cfg :rope-base)
                    (plist-get cfg :tied-head)))

    ;; --- the three independent comparisons, per sampled row ---------------
    (let ((lane-bad nil) (scale-bad nil) (value-bad nil) (n 0) (cols 0))
      (dolist (s samples)
        (let* ((role (plist-get s :role))
               (layer (plist-get s :layer))
               (row (plist-get s :row))
               (kind (plist-get s :kind))
               (tn (nl-llm-weights-tensor wts role layer))
               (tag (format "%s[%d] row %d" role layer row)))
          (setq n (1+ n) cols (+ cols (plist-get s :cols)))
          (unless (equal (plist-get tn :kind) kind)
            (push (format "%s kind %S != %S" tag (plist-get tn :kind) kind)
                  lane-bad))
          (when (equal kind "int8x4")
            (let ((got (nl-llm-weights-lanes wts tn row))
                  (want (plist-get s :lanes)))
              (unless (equal (append got nil) want)
                (push (format "%s lanes differ (first %S vs %S)" tag
                              (and (> (length got) 0) (aref got 0))
                              (car want))
                      lane-bad)))
            (let ((got (aref (nl-llm-weights-scales wts tn) row))
                  (want (plist-get s :scale)))
              (unless (= got want)
                (push (format "%s scale %S != %S" tag got want) scale-bad))))
          (let ((got (nl-llm-weights-row wts tn row))
                (want (plist-get s :values)))
            (unless (equal (append got nil) want)
              (let ((k (cl-loop for a across got for b in want for i from 0
                                unless (= a b) return i)))
                (push (format "%s values differ at %S (%S vs %S)" tag k
                              (and k (aref got k)) (and k (nth k want)))
                      value-bad))))))

      (wl--ck "unpacked int8 lanes == exporter" (null lane-bad)
              (if lane-bad (car (nreverse lane-bad))
                (format "%d rows" n)))
      (wl--ck "per-row scales == exporter" (null scale-bad)
              (if scale-bad (car (nreverse scale-bad)) ""))
      (wl--ck "dequantized values == exporter (exact, f64)" (null value-bad)
              (if value-bad (car (nreverse value-bad))
                (format "%d values over %d rows" cols n))))

    ;; f32 tensors come back as photon-tensors of the right shape.
    (let* ((tn (nl-llm-weights-tensor wts :ln1g 0))
           (tensor (nl-llm-weights-f32-tensor wts tn)))
      (require 'photon-tensor)
      (wl--ck "f32 tensor loads as a photon-tensor"
              (equal (photon-tensor-shape tensor) (plist-get tn :shape))
              (format "%S" (photon-tensor-shape tensor))))

    ;; Refusing to dequantize a large tensor wholesale is the memory contract,
    ;; so it is a behaviour under test rather than a comment.
    (wl--ck "int8 tensors refuse wholesale dequantization"
            (condition-case err
                (progn (nl-llm-weights-f32-tensor
                        wts (nl-llm-weights-tensor wts :wte))
                       nil)
              (error (and (string-match-p "int8x4"
                                          (error-message-string err)) t))))

    ;; Raw bytes are handed over verbatim, at the length the header declares --
    ;; this is what an uploader sends to a GPU buffer without touching a float.
    (let* ((tn (nl-llm-weights-tensor wts :wq 0))
           (bytes (nl-llm-weights-bytes wts tn))
           (shape (plist-get tn :shape)))
      (wl--ck "raw bytes match the declared extent"
              (and (= (length bytes) (plist-get tn :nbytes))
                   (not (multibyte-string-p bytes))
                   (= (plist-get tn :nbytes)
                      (* (car shape) (plist-get tn :words) 4)))
              (format "%d bytes, %d rows x %d words x 4"
                      (length bytes) (car shape) (plist-get tn :words))))

    ;; Out-of-range access must signal, not read a neighbouring row.
    (wl--ck "row index is bounds-checked"
            (condition-case err
                (progn (nl-llm-weights-lanes
                        wts (nl-llm-weights-tensor wts :wq 0) 999999)
                       nil)
              (error (and (string-match-p "outside" (error-message-string err))
                          t))))

    (princ (format "\n%s: %d failure(s)\n"
                   (if (zerop wl--fail) "weights-load OK" "weights-load")
                   wl--fail))
    (when (> wl--fail 0) (kill-emacs 1))))
