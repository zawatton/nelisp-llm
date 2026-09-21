;;; soft-vocab-report.el --- how big a vocabulary the soft targets need  -*- lexical-binding: t; -*-

;; The student's vocabulary is the first thing a soft-target run forces a
;; decision on, and it is a measurement, not a preference.  Two questions, and
;; the second is the one that is usually skipped:
;;
;;   1. How much of the corpus does a vocabulary of N tokens serve?  Reported
;;      three ways -- mass, positions whose sampled token is representable, and
;;      positions whose whole top-k is -- because only the second and third
;;      decide whether a position can be trained on at all.
;;
;;   2. How fast does the distinct-token count grow with the corpus?  A
;;      vocabulary chosen from nine examples is not a vocabulary chosen for a
;;      thousand, and the growth curve says by how much.  Measured on prefixes
;;      of the dataset, which is free.
;;
;; Usage: emacs -Q --batch -L lisp -L <photon> -l tools/soft-vocab-report.el
;; Env: NL_LLM_SOFT_DATA (default build/distilled-soft.eld)

(require 'cl-lib)
(require 'nl-llm-distill)
(require 'nl-llm-distill-soft)

(let* ((path (or (getenv "NL_LLM_SOFT_DATA") "build/distilled-soft.eld"))
       (data (nl-llm-distill-read path))
       (examples (append (plist-get data :examples) nil))
       (vocab (nl-llm-distill-soft-vocabulary examples))
       (positions (apply #'+ 0 (mapcar (lambda (e) (length (plist-get e :tokens)))
                                       examples))))
  (message "%s: teacher %s top-%s, %d example(s), %d position(s), %d distinct"
           (file-name-nondirectory path)
           (plist-get data :teacher) (plist-get data :top-k)
           (length examples) positions (length vocab))

  (message "\n  what a vocabulary of N tokens serves")
  (message "  %-8s %-10s %-12s %s" "N" "mass" "sampled" "complete")
  (dolist (n '(256 512 1024 2048 4096 8192))
    (when (<= n (max 8192 (length vocab)))
      (let ((cov (nl-llm-distill-soft-position-coverage examples n)))
        (message "  %-8d %-10.4f %-12.4f %.4f"
                 n (nl-llm-distill-soft-coverage vocab n)
                 (plist-get cov :sampled) (plist-get cov :complete)))))

  ;; The sampled tokens alone.  A renormalised target over a kept subset needs
  ;; only the *sampled* token to be representable, so this distribution -- much
  ;; narrower than the union with every alternative -- is what a pruned
  ;; vocabulary would actually have to hold.
  (let ((sampled (let ((h (make-hash-table :test 'equal)))
                   (dolist (e examples)
                     (dolist (position (plist-get e :tokens))
                       (puthash (plist-get position :token)
                                (1+ (gethash (plist-get position :token) h 0))
                                h)))
                   (let (out) (maphash (lambda (k v) (push (cons k v) out)) h)
                        (sort out (lambda (a b) (> (cdr a) (cdr b))))))))
    (message "\n  sampled tokens only: %d distinct of %d position(s)"
             (length sampled) positions)
    (message "  %-8s %s" "N" "positions whose sampled token is in the top N")
    (dolist (n '(256 512 1024 2048 4096))
      (when (< n (length sampled))
        (let* ((keep (apply #'+ 0 (mapcar #'cdr (cl-subseq sampled 0 n)))))
          (message "  %-8d %.4f" n (/ (float keep) positions)))))
    (message "\n  how the sampled-token count grows")
    (message "  %-10s %-12s %-12s %s" "examples" "positions" "distinct" "new")
    (let ((previous 0))
      (dotimes (i (length examples))
        (let* ((prefix (cl-subseq examples 0 (1+ i)))
               (h (make-hash-table :test 'equal))
               (p 0))
          (dolist (e prefix)
            (dolist (position (plist-get e :tokens))
              (setq p (1+ p))
              (puthash (plist-get position :token) t h)))
          (message "  %-10d %-12d %-12d %+d" (1+ i) p (hash-table-count h)
                   (- (hash-table-count h) previous))
          (setq previous (hash-table-count h))))))

  ;; Growth.  If distinct tokens still climb steeply at the end of the corpus,
  ;; a vocabulary sized here will not hold for a corpus a hundred times larger,
  ;; and the honest answer is to size it from a larger sample rather than to
  ;; extrapolate a curve from nine points.
  (message "\n  how the distinct count grows with the corpus")
  (message "  %-10s %-12s %-12s %s" "examples" "positions" "distinct" "per new example")
  (let ((previous 0))
    (dotimes (i (length examples))
      (let* ((k (1+ i))
             (prefix (cl-subseq examples 0 k))
             (v (length (nl-llm-distill-soft-vocabulary prefix)))
             (p (apply #'+ 0 (mapcar (lambda (e) (length (plist-get e :tokens)))
                                     prefix))))
        (message "  %-10d %-12d %-12d %+d" k p v (- v previous))
        (setq previous v)))))

;;; soft-vocab-report.el ends here
