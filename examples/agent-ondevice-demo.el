;;; agent-ondevice-demo.el --- the self-improvement loop, closed entirely on the GPU  -*- lexical-binding: t; -*-
;; The whole loop now runs on-device.  One model with two views that SHARE weight
;; tensors: a CPU view rolls out actions, an nlga GPU graph (built from the same
;; tensors) trains on the wins, and `nlga-readback' copies the GPU-trained weights
;; straight back into the shared tensors -- so the CPU rollout immediately decodes
;; with the GPU-trained weights.  rollout (CPU) -> reward -> train (GPU) -> readback
;; -> repeat.  The CPU rollout success rate rising, with the weights provably
;; changed only by the GPU training, shows the transfer closes the loop.
;; Needs a Vulkan device.  (docs/design/05-agent-harness.org)
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l examples/agent-ondevice-demo.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-agent)
(require 'nl-llm-agent-model)
(require 'nl-llm-agent-improve)
(require 'nl-llm-gpu)
(require 'nl-llm-agent-ondevice)

(random "p5-ondevice-demo-seed")

(princ "=== self-improvement loop closed on the GPU (weight transfer) ===\n")
(unless (nl-llm-gpu-enable) (princ "SKIP: no Vulkan device\n") (kill-emacs 0))
(let* ((grammar (nl-llm-agent-grammar-message 1 "abcd"))
       (target "a")
       (reward (lambda (e) (if (and (string-match "(message \"\\([a-d]\\)\")" e) (equal (match-string 1 e) target)) 1.0 0.0)))
       (ctx (nl-llm-agent-ondevice-new 16 16 1 1 24 0.1))
       (wh (plist-get (plist-get ctx :cpu) :wh))
       (w0 (aref (photon-tensor-data (pav-value wh)) 0)))
  (princ (format "task: emit (message \"x\") with x=%S; CPU rolls out, GPU trains, readback transfers weights back.\n\n" target))
  (princ (format "before: a CPU-rollout action = %S\n\n" (car (nl-llm-agent-p5-rollout (plist-get ctx :cpu) grammar 1.0))))
  (let ((rates (nl-llm-agent-improve-ondevice ctx grammar reward
                :rounds 6 :rollouts 24 :epochs 4 :temp 1.0 :eval-n 24
                :trace (lambda (r n rate) (princ (format "  round %d: %2d replay -> GPU train -> readback -> CPU success %3.0f%%\n" r n (* 100 rate)))))))
    (let ((w1 (aref (photon-tensor-data (pav-value wh)) 0)))
      (princ (format "\nCPU-rollout success: %s\n" (mapconcat (lambda (x) (format "%.0f%%" (* 100 x))) rates " -> ")))
      (princ (format "after : a CPU-rollout action = %S\n" (car (nl-llm-agent-p5-rollout (plist-get ctx :cpu) grammar 0.15))))
      (princ (format "\nshared head weight wh[0]: %.4f -> %.4f  (changed only by GPU training, via readback)\n" w0 w1))
      (nl-llm-agent-ondevice-free ctx) (nl-llm-gpu-disable)
      (let ((ok (and (/= w0 w1) (> (apply #'max rates) (+ (car rates) 0.25)))))
        (princ (format "\nloop closed on-device (transfer + rollout improved): %s  (%.0f%% -> peak %.0f%%)\n"
                       (if ok "YES" "NO") (* 100 (car rates)) (* 100 (apply #'max rates))))
        (kill-emacs (if ok 0 1))))))
;;; agent-ondevice-demo.el ends here
