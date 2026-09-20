;;; bonsai-zero-layer.el --- which of the two rotations is the model's -*- lexical-binding: t -*-

;; The embedding and the head sit at opposite ends of the same basis: the file
;; folds the inverse transform into `token_embd.weight' and the forward one
;; into `output.weight', so under the right transform a token's embedding, run
;; through the final norm and the head with no blocks in between, scores that
;; token far above chance.  Under the wrong one it scores it like any other
;; token, because the wrong transform is still orthogonal and still produces a
;; perfectly well-formed vector pointing somewhere else.
;;
;; Chance is rank 124160 of 248320.  This needs the head and nothing else, so
;; it costs seconds where running all 64 blocks costs a quarter of an hour.

(require 'nl-llm-bonsai)
(require 'nl-llm-weights-gpu)
(require 'nelisp-gpu-server)
(require 'cl-lib)

(defun bzl--rank (lg id)
  (let ((r 1) (v (aref lg id)))
    (dotimes (i (length lg)) (when (> (aref lg i) v) (setq r (1+ r))))
    r))

(defun bzl-run ()
  (let* ((path (or (getenv "NL_BONSAI_WTS") "build/bonsai/wts-3.bin"))
         (ids (mapcar #'string-to-number
                      (split-string (or (getenv "NL_BONSAI_TOKENS")
                                        "760 6511 314 9338 369 11751 25358 21047")
                                    "[ ,]+" t))))
    (nelisp-gpu-server-start)
    (unwind-protect
        (dolist (mode '((nil . "forward") (t . "inverse") (:none . "no rotation")))
          (let* ((invert (car mode))
                 (sess (nl-llm-bonsai-open path (and (not (eq invert :none)) invert)))
                 (cfg (plist-get sess :cfg))
                 (vocab (plist-get cfg :vocab))
                 (head (nl-llm-bonsai-head sess))
                 (tbl (make-hash-table :test 'eq))
                 (ranks nil))
            (puthash head (nl-llm-wgpu-upload-lin head) tbl)
            (unwind-protect
                (let ((nl-llm-wgpu--transposes tbl)
                      (nl-llm-bonsai-apply-fn #'nl-llm-wgpu-apply-resident)
                      (dim (plist-get cfg :dim)))
                  (dolist (id ids)
                    (let* ((e (if (eq invert :none)
                                  (nl-llm-weights-embed (plist-get sess :wts) id)
                                (nl-llm-bonsai-embed sess id)))
                           (lg (nl-llm-bonsai-logits sess e 1 0))
                           (r (bzl--rank lg id)))
                      (push r ranks)
                      (message "  %-12s token %6d  self rank %7d / %d  logit %8.3f  argmax %6d"
                               (cdr mode) id r vocab (aref lg id)
                               (car (nl-llm-bonsai-argmax lg))))))
              (nl-llm-wgpu-free-transposes tbl))
            (let* ((rs (nreverse ranks))
                   (med (nth (/ (length rs) 2) (sort (copy-sequence rs) #'<))))
              (message "  %-12s median self rank %d  (chance %d)"
                       (cdr mode) med (/ vocab 2))
              (message ""))))
      (nelisp-gpu-server-stop))
    (message "DONE")))

(bzl-run)
