;;; nl-llm-gpu.el --- GPU acceleration for the modern block (train + decode)  -*- lexical-binding: t; -*-

;; Routes the modern block's heavy linear algebra through the nelisp-gpu
;; SPIR-V/Vulkan kernels via the photon-tensor-gpu function-cell swap.  The
;; modern block uses `photon-tensor-linear' / `photon-tensor-matmul' /
;; `photon-tensor-softmax-rows' directly (forward) and through
;; `photon-autograd-*' (training), so enabling the backend accelerates both
;; the projections / expert FFNs and the attention score/context matmuls.
;;
;; Training caveat handled here: weights are updated in place by SGD, but the
;; nelisp-gpu resident-weight cache is keyed by vector identity and so cannot
;; see an in-place mutation -- it would keep using the first upload.  After
;; each SGD step the caller must `nl-llm-gpu-invalidate' the parameters so the
;; next forward re-uploads the current weights.  (For inference, where weights
;; are constant, residency is correct and uploads each weight exactly once.)
;;
;;   (nl-llm-gpu-enable)
;;   ... forward / generate ...                 ; weights constant -> resident
;;   ... or training loop:
;;       (photon-autograd-backward loss)
;;       (photon-autograd-sgd params lr)
;;       (nl-llm-gpu-invalidate params)         ; <- refresh GPU copies
;;   (nl-llm-gpu-disable)

;;; Code:

(require 'photon-tensor)
(require 'photon-autograd)
(require 'photon-tensor-gpu)   ; pulls in nelisp-gpu-server + the op swaps

;; Point the server client at the sibling nelisp-gpu binary, independent of
;; the process `default-directory' (photon-tensor-gpu only sets the vkrun
;; path, not the persistent-server binary).
(setq nelisp-gpu-server-bin
      (expand-file-name
       "../../nelisp-gpu/host/vkserver"
       (file-name-directory (or load-file-name buffer-file-name default-directory))))

;;;###autoload
(defun nl-llm-gpu-available-p ()
  "Return non-nil if the persistent GPU server is currently running."
  (nelisp-gpu-server-up-p))

;;;###autoload
(defun nl-llm-gpu-enable ()
  "Start the persistent GPU server and route the heavy photon-tensor ops
through the nelisp-gpu kernels.  Returns 'gpu on success, nil if the server
could not be started (no server binary / no Vulkan device) -- callers should
fall back to the CPU path on nil."
  (when (condition-case nil
            (progn (nelisp-gpu-server-start) (nelisp-gpu-server-up-p))
          (error nil))
    (photon-tensor-use-gpu-backend)
    'gpu))

;;;###autoload
(defun nl-llm-gpu-disable ()
  "Restore the pure-elisp photon-tensor ops and stop the GPU server."
  (photon-tensor-use-cpu-backend)
  (nelisp-gpu-server-stop)
  'cpu)

;;;###autoload
(defun nl-llm-gpu-invalidate (params)
  "Invalidate the resident GPU copies of PARAMS' weight/bias vectors.
PARAMS is a list of `pav' parameters (as passed to `photon-autograd-sgd').
Call this right after each in-place SGD step so the next forward re-uploads
the updated weights instead of reusing the stale GPU buffers."
  (when (nelisp-gpu-server-up-p)
    (dolist (p params)
      (nelisp-gpu-server-invalidate (photon-tensor-data (pav-value p))))))

(provide 'nl-llm-gpu)
;;; nl-llm-gpu.el ends here
