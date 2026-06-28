;;; agent-model-demo.el --- P4: a real nelisp-llm model driving the agent  -*- lexical-binding: t; -*-
;; Wires an actual nelisp-llm model (CPU transformer decode + KV cache) in as the
;; agent policy, with constrained decoding under an action grammar.  The model
;; here is UNTRAINED (random weights), so the content it fills in is gibberish --
;; but the grammar guarantees every step is a structurally valid, parseable,
;; executable action.  That is the Phase 4 result: constrained decoding turns any
;; model, however weak, into a structurally-valid agent; Phase 5 then trains the
;; model on its own successful trajectories so the content becomes useful too
;; (docs/design/05-agent-harness.org).  No GPU needed.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l examples/agent-model-demo.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-agent)
(require 'nl-llm-agent-model)

(defun amd--t (shape seed sc) (let ((n 1)) (dolist (d shape) (setq n (* n d)))
  (photon-tensor shape (let ((v (make-vector n 0.0)) (i 0))
    (while (< i n) (aset v i (* sc 2.0 (- (/ (float (mod (+ (* (1+ i) 2654435761) (* (1+ seed) 40503)) 65536)) 65536.0) 0.5))) (setq i (1+ i))) v))))
(defun amd--ones (n) (photon-tensor (list n) (make-vector n 1.0)))

(let* ((dim 32) (heads 4) (kvh 2) (ff 48) (vocab nl-llm-agent-char-vocab) (hd (/ dim heads)) (kvdim (* kvh hd))
       (mkblk (lambda (s0) (list :ln1g (amd--ones dim) :wq (amd--t (list dim dim) (+ s0 1) 0.4) :bq (amd--t (list dim) (+ s0 11) 0.1)
                                 :wk (amd--t (list kvdim dim) (+ s0 2) 0.4) :bk (amd--t (list kvdim) (+ s0 12) 0.1)
                                 :wv (amd--t (list kvdim dim) (+ s0 3) 0.4) :bv (amd--t (list kvdim) (+ s0 13) 0.1)
                                 :wo (amd--t (list dim dim) (+ s0 4) 0.4) :bo (amd--t (list dim) (+ s0 14) 0.1) :ln2g (amd--ones dim)
                                 :wg (amd--t (list ff dim) (+ s0 5) 0.4) :bg (amd--t (list ff) (+ s0 15) 0.1)
                                 :wu (amd--t (list ff dim) (+ s0 6) 0.4) :bu (amd--t (list ff) (+ s0 16) 0.1)
                                 :wd (amd--t (list dim ff) (+ s0 7) 0.4) :bd (amd--t (list dim) (+ s0 17) 0.1))))
       (model (list :blocks (list (funcall mkblk 100) (funcall mkblk 200))
                    :wte (amd--t (list vocab dim) 1 0.4) :lnfg (amd--ones dim) :bh (amd--t (list vocab) 19 0.1)
                    :dim dim :heads heads :kvh kvh))
       (policy (nl-llm-agent-model-policy model (nl-llm-agent-grammar-message 12) 512))
       (valid 0) (total 0))
  (princ "=== P4: an (untrained) nelisp-llm model drives the agent via constrained decoding ===\n")
  (princ "model: CPU transformer decode, char vocab=96, 2 blocks; grammar forces a valid Elisp action.\n\n")
  (let ((res (nl-llm-agent-run
              "produce an action" policy
              :system "Emit one action." :max-steps 3 :workdir default-directory
              :trace (lambda (step role content)
                       (when (and (> step 0) (memq role '(assistant observation)))
                         (when (eq role 'assistant)
                           (setq total (1+ total))
                           (when (eq (car (nl-llm-agent--parse content)) 'elisp) (setq valid (1+ valid))))
                         (princ (format "[step %d %s] %s\n" step role
                                        (string-trim (replace-regexp-in-string "\n" "\\\\n" content)))))))))
    (princ (format "\nstructural validity: %d/%d steps were valid, parseable, executable Elisp actions\n" valid total))
    (princ "(content is gibberish because the model is untrained; the grammar guarantees the STRUCTURE.\n")
    (princ " Phase 5 trains the model on successful trajectories so the content becomes useful.)\n")
    (kill-emacs (if (and (= valid total) (> total 0)) 0 1))))
;;; agent-model-demo.el ends here
