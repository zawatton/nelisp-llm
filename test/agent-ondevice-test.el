;;; agent-ondevice-test.el --- the self-improvement loop closed on the GPU  -*- lexical-binding: t; -*-
;; The whole loop runs on-device: rollout on the CPU view, train the SHARED nlga
;; graph on the GPU, readback the trained weights into the shared tensors, repeat.
;; The CPU rollout success rate rises -- and the only thing that changed its weights
;; is the GPU training fed back through `nlga-readback', so the rise proves the
;; weight transfer closes the loop.  Skips (exit 0) without a Vulkan device.
;;   emacs -Q --batch -L lisp -L ../nelisp-photon/lisp -l test/agent-ondevice-test.el
(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'cl-lib)
(require 'photon-tensor)
(require 'nl-llm-agent)
(require 'nl-llm-agent-model)
(require 'nl-llm-agent-improve)
(require 'nl-llm-gpu)
(require 'nl-llm-agent-ondevice)

(defvar od--fail 0)
(defun od--ck (name ok &optional extra)
  (princ (format "%-54s %s  %s\n" name (if ok "PASS" (progn (setq od--fail (1+ od--fail)) "FAIL")) (or extra ""))))

(random "p5-ondevice-seed")

(if (not (nl-llm-gpu-enable))
    (od--ck "on-device self-improvement loop [SKIPPED: no GPU]" t)
  (let* ((grammar (nl-llm-agent-grammar-message 1 "abcd"))
         (target "a")
         (reward (lambda (e) (if (and (string-match "(message \"\\([a-d]\\)\")" e) (equal (match-string 1 e) target)) 1.0 0.0)))
         (ctx (nl-llm-agent-ondevice-new 16 16 1 1 24 0.1))   ; dim ff heads nblocks seq lr
         ;; capture a weight value before training, to confirm the transfer mutates it
         (w0 (aref (photon-tensor-data (pav-value (plist-get (plist-get ctx :cpu) :wh))) 0))
         (rates (nl-llm-agent-improve-ondevice ctx grammar reward
                  :rounds 6 :rollouts 24 :epochs 4 :temp 1.0 :eval-n 24
                  :trace (lambda (r n rate) (princ (format "  round %d: %2d replay -> CPU-rollout success %3.0f%% (after GPU train + readback)\n" r n (* 100 rate))))))
         (w1 (aref (photon-tensor-data (pav-value (plist-get (plist-get ctx :cpu) :wh))) 0)))
    (nl-llm-agent-ondevice-free ctx)
    (nl-llm-gpu-disable)
    (princ (format "CPU-rollout success: %s\n" (mapconcat (lambda (x) (format "%.0f%%" (* 100 x))) rates " -> ")))
    (od--ck "GPU training mutated the shared CPU weights (transfer happened)" (/= w0 w1)
            (format "wh[0] %.4f -> %.4f" w0 w1))
    (od--ck "loop closed: CPU-rollout success rose after GPU train+readback"
            (> (apply #'max rates) (+ (car rates) 0.25))
            (format "%.0f%% -> peak %.0f%%" (* 100 (car rates)) (* 100 (apply #'max rates))))
    (od--ck "reached well above the random baseline (2x+)" (> (apply #'max rates) 0.55)
            (format "peak %.0f%%" (* 100 (apply #'max rates))))))

(princ (format "NL-LLM-AGENT-ONDEVICE %s (%d failures)\n" (if (= od--fail 0) "ALL-PASS" "HAS-FAILURES") od--fail))
(kill-emacs (if (= od--fail 0) 0 1))
;;; agent-ondevice-test.el ends here
