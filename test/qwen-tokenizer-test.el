;;; qwen-tokenizer-test.el --- donor token-id parity  -*- lexical-binding: t; -*-
;; Run: emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/qwen-tokenizer-test.el
;;
;; Doc 08 Phase 1, Verification checks 1 and 2.  Every fixture id sequence comes
;; from the reference `tokenizers' library (tools/qwen-tokenizer-fixtures.py),
;; so a pass means "identical to the donor", not "self-consistent".
;;
;; The last check is a NEGATIVE CONTROL: it perturbs one merge rank and demands
;; that the comparison then fails.  Without it, a parity suite that silently
;; compared nothing -- an empty fixture list, an encode that returned the
;; expectation, a loader that no-oped -- would report all green.
;;
;; The table and fixtures are donor-derived and gitignored; the suite skips
;; rather than fails when they are absent, so a fresh clone is not red.

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'nl-llm-qwen-tokenizer)

(defvar qt--fail 0)
(defvar qt--table (expand-file-name "build/donor/qwen3-0.6b/tokenizer.bin"))
(defvar qt--fixtures (expand-file-name "test/fixtures/qwen-tokenizer.eld"))

(defun qt--ck (name ok &optional extra)
  (princ (format "%-46s %s  %s\n" name
                 (if ok "PASS" (progn (setq qt--fail (1+ qt--fail)) "FAIL"))
                 (or extra ""))))

(defun qt--read-fixtures (path)
  "Read (CODEPOINTS . IDS) entries from PATH as (TEXT . IDS) pairs."
  (with-temp-buffer
    (let ((coding-system-for-read 'utf-8)) (insert-file-contents path))
    (goto-char (point-min))
    (mapcar (lambda (entry)
              (cons (if (car entry) (apply #'string (car entry)) "")
                    (cdr entry)))
            (read (current-buffer)))))

(defun qt--show (text)
  "Return a short one-line rendering of TEXT for a failure message."
  (let ((s (prin1-to-string text)))
    (if (> (length s) 40) (concat (substring s 0 40) "...") s)))

(if (not (and (file-readable-p qt--table) (file-readable-p qt--fixtures)))
    (princ (format "SKIP: donor table or fixtures missing\n  %s\n  %s\n\
  regenerate with:  make qwen-tokenizer-table\n" qt--table qt--fixtures))

  (let* ((t0 (float-time))
         (tok (nl-llm-qwen-tok-load qt--table))
         (load-secs (- (float-time) t0))
         (cases (qt--read-fixtures qt--fixtures)))

    (qt--ck "table loads" (nl-llm-qwen-tok-p tok)
            (format "%d merges, %d added, vocab %d, %.2fs"
                    (hash-table-count (nl-llm-qwen-tok-merges tok))
                    (length (nl-llm-qwen-tok-added tok))
                    (nl-llm-qwen-tok-vocab-size tok)
                    load-secs))
    (qt--ck "fixtures non-empty" (> (length cases) 0)
            (format "%d cases" (length cases)))

    ;; 1. id parity against the reference, case by case.
    (let ((bad nil) (n 0))
      (dolist (c cases)
        (setq n (1+ n))
        (let ((got (nl-llm-qwen-tok-encode tok (car c))))
          (unless (equal got (cdr c))
            (push (format "\n    %s\n      want %S\n      got  %S"
                          (qt--show (car c)) (cdr c) got)
                  bad))))
      (qt--ck "encode == reference ids"
              (null bad)
              (if bad
                  (format "%d/%d mismatched%s" (length bad) n
                          (apply #'concat (nreverse bad)))
                (format "%d/%d cases" n n))))

    ;; 2. decode round-trips.  NFC folding means decode(encode(s)) can differ
    ;; from s, so the invariant is against the reference's own decode: decoding
    ;; the reference ids must reproduce our decode of the same ids.
    (let ((bad nil))
      (dolist (c cases)
        (let* ((ids (cdr c))
               (text (nl-llm-qwen-tok-decode tok ids)))
          (unless (equal ids (nl-llm-qwen-tok-encode tok text))
            (push (qt--show (car c)) bad))))
      (qt--ck "decode then encode is stable" (null bad)
              (if bad (format "%d unstable: %s" (length bad)
                              (mapconcat #'identity (nreverse bad) " "))
                "")))

    ;; 3. every byte value is representable, so no input can be unencodable.
    (let ((bad nil))
      (dotimes (b 256)
        (unless (aref (nl-llm-qwen-tok-byte->id tok) b) (push b bad)))
      (qt--ck "byte alphabet complete" (null bad)
              (if bad (format "missing %S" bad) "256/256")))

    ;; 3b. the pinned category ranges reproduce the reference pre-tokenizer's
    ;; own verdict, sampled independently of the exporter's full sweep, so the
    ;; probe -> ranges -> binary -> binary-search path is checked end to end.
    ;;
    ;; Emacs's `get-char-code-property' is deliberately NOT the authority: the
    ;; reference library tracks its own Unicode version and disagrees with both
    ;; Emacs and Python on newly assigned code points.  Reading categories from
    ;; the host would make token ids depend on the host's Unicode version, with
    ;; no error when they shift.  The host skew is reported, not asserted.
    (let ((cats (expand-file-name "test/fixtures/qwen-tokenizer-cats.eld")))
      (if (not (file-readable-p cats))
          (qt--ck "pinned categories" t "fixture absent, skipped")
        (let ((flat (with-temp-buffer
                      (let ((coding-system-for-read 'utf-8-unix))
                        (insert-file-contents cats))
                      (goto-char (point-min))
                      (read (current-buffer))))
              (bad nil) (n 0) (skew 0))
          (while flat
            (let* ((cp (pop flat))
                   (want (pop flat))
                   (got (cond ((nl-llm-qwen-tok-letter-p tok cp) 1)
                              ((nl-llm-qwen-tok-number-p tok cp) 2)
                              (t 0)))
                   (cat (get-char-code-property cp 'general-category))
                   (init (if cat (aref (symbol-name cat) 0) ?C))
                   (host (cond ((eq init ?L) 1) ((eq init ?N) 2) (t 0))))
              (setq n (1+ n))
              (unless (eq got want)
                (push (format "U+%04X want %d got %d" cp want got) bad))
              (unless (eq host want) (setq skew (1+ skew)))))
          (qt--ck "pinned categories == reference"
                  (null bad)
                  (if bad (format "%d of %d disagree: %s" (length bad) n
                                  (mapconcat #'identity
                                             (cl-subseq (nreverse bad) 0
                                                        (min 6 (length bad)))
                                             " "))
                    (format "%d code points" n)))
          (princ (format "%-46s %s  %d of %d code points\n"
                         "  (note) host Unicode differs from donor" "----"
                         skew n)))))

    ;; 4. pre-tokenizer covers its input exactly (no dropped or duplicated text).
    (let ((bad nil))
      (dolist (c cases)
        (let ((text (ucs-normalize-NFC-string (car c))))
          (unless (equal text (apply #'concat
                                     (nl-llm-qwen-tok-pretokenize tok text)))
            (push (qt--show (car c)) bad))))
      (qt--ck "pre-tokenizer pieces rejoin" (null bad)
              (if bad (mapconcat #'identity (nreverse bad) " ") "")))

    ;; 5. NEGATIVE CONTROL.  Perturb one merge that a fixture actually uses and
    ;; require the parity check to notice.  A green run here means check 1 is
    ;; incapable of failing and must not be trusted.
    (let* ((merges (nl-llm-qwen-tok-merges tok))
           (probe "hello world")
           (before (nl-llm-qwen-tok-encode tok probe))
           (key nil))
      ;; Find a merge the probe's first byte pair uses, then break its result.
      (let ((bytes (encode-coding-string probe 'utf-8))
            (b->i (nl-llm-qwen-tok-byte->id tok)))
        (setq key (+ (* (aref b->i (aref bytes 0)) 262144)
                     (aref b->i (aref bytes 1)))))
      (let ((saved (gethash key merges)))
        (if (null saved)
            (qt--ck "negative control" nil
                    "probe uses no merge; pick a different probe")
          (puthash key (+ (* 0 262144) (mod (+ 1 saved) 262144)) merges)
          (let ((after (nl-llm-qwen-tok-encode tok probe)))
            (puthash key saved merges)
            (qt--ck "negative control: broken merge is detected"
                    (not (equal before after))
                    (format "%S -> %S" before after))
            (qt--ck "negative control: table restored"
                    (equal before (nl-llm-qwen-tok-encode tok probe)))))))

    ;; 5b. bulk parity over real corpora.  71 hand-picked cases prove the
    ;; branches; whole files prove nothing drifts over thousands of tokens of
    ;; prose, Elisp and Japanese.  Each entry is pinned to a digest of its
    ;; source text, so editing a corpus file makes the fixture stale and loud
    ;; instead of quietly comparing against ids from older text.
    (let ((bulk (expand-file-name "test/fixtures/qwen-tokenizer-bulk.eld")))
      (if (not (file-readable-p bulk))
          (qt--ck "bulk parity" t "fixture absent, skipped")
        (dolist (entry (with-temp-buffer
                         (let ((coding-system-for-read 'utf-8-unix))
                           (insert-file-contents bulk))
                         (goto-char (point-min))
                         (read (current-buffer))))
          (let* ((rel (nth 0 entry))
                 (digest (nth 1 entry))
                 (want (nth 2 entry))
                 (path (expand-file-name rel))
                 (text (and (file-readable-p path)
                            (with-temp-buffer
                              (let ((coding-system-for-read 'utf-8-unix))
                                (insert-file-contents path))
                              (buffer-string)))))
            (cond
             ((null text) (qt--ck (format "bulk %s" rel) nil "file missing"))
             ((not (equal digest (secure-hash
                                  'sha256
                                  (encode-coding-string text 'utf-8-unix))))
              (qt--ck (format "bulk %s" rel) nil
                      "source changed since the fixture was generated"))
             (t
              (let* ((got (nl-llm-qwen-tok-encode tok text))
                     (ok (equal got want))
                     (at (unless ok
                           (cl-loop for a in got for b in want for k from 0
                                    unless (equal a b) return k))))
                (qt--ck (format "bulk %s" rel) ok
                        (if ok (format "%d ids" (length want))
                          (format "%d vs %d ids, first diff at %s"
                                  (length got) (length want) at))))))))))

    ;; 6. throughput, so a later optimisation has a baseline to beat.
    (let* ((text (mapconcat #'car cases ""))
           (t1 (float-time))
           (ids (nl-llm-qwen-tok-encode tok text))
           (secs (- (float-time) t1)))
      (qt--ck "throughput" t
              (format "%d chars -> %d ids in %.3fs (%.0f chars/s)"
                      (length text) (length ids) secs
                      (if (> secs 0) (/ (length text) secs) 0))))

    (princ (format "\n%s: %d failure(s)\n"
                   (if (zerop qt--fail) "qwen-tokenizer OK" "qwen-tokenizer")
                   qt--fail))
    (when (> qt--fail 0) (kill-emacs 1))))
