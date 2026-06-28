;;; nl-llm-gpu-train.el --- on-device (resident) training step  -*- lexical-binding: t; -*-

;; On-device training: keep the weights *and* the optimiser update on the GPU.
;; Weights, inputs and targets are uploaded once to resident GPU buffers; each
;; step runs the whole forward + backward + SGD as ONE fused command buffer
;; that updates the resident weight buffers IN PLACE (the `sgd' kernel writes
;; back into the resident slots).  Nothing weight-sized crosses the host
;; boundary per step, so the per-step cost no longer includes re-encoding and
;; re-uploading the mutated weights -- the bottleneck that made the host-driven
;; GPU training path only break-even with the CPU.
;;
;; This is demonstrated on a 2-layer MLP (linear -> gelu -> linear, MSE loss),
;; the shape of a transformer FFN -- the dominant weight cost in a real model.
;; `nl-llm-gpu-mlp-train' trains it fully on-device and writes the trained
;; weights back to the host tensors at the end.

;;; Code:

(require 'photon-tensor)
(require 'nl-llm-gpu)   ; pulls in nelisp-gpu-server + the backend, sets bin path

(defsubst nl-llm-gpu--groups (n) (/ (+ n 63) 64))

;;;###autoload
(defun nl-llm-gpu-mlp-train (x w1 b1 w2 b2 target lr steps)
  "Train a 2-layer MLP fully on the GPU and return the list of per-step losses.
X (seq x in), W1 (hid x in), B1 (hid), W2 (out x hid), B2 (out) and TARGET
\(seq x out) are plain photon-tensors.  Computes y = gelu(X W1^T + b1) W2^T + b2
and minimises mean-squared error to TARGET with plain SGD at rate LR for STEPS
steps.  Weights are kept resident on the GPU and updated in place on-device;
the trained weights are read back into W1/B1/W2/B2 at the end.  The server must
be running (see `nl-llm-gpu-enable')."
  (unless (nelisp-gpu-server-up-p) (error "nl-llm-gpu-mlp-train: GPU server not running"))
  (let* ((seq (car (photon-tensor-shape x)))   (in  (nth 1 (photon-tensor-shape x)))
         (hid (car (photon-tensor-shape w1)))  (out (car (photon-tensor-shape w2)))
         (ntot (* seq out)) (invn (/ 1.0 (float ntot)))
         (g #'nl-llm-gpu--groups)
         ;; upload everything resident once
         (hx   (nelisp-gpu-server-upload (photon-tensor-data x)))
         (hw1  (nelisp-gpu-server-upload (photon-tensor-data w1)))
         (hb1  (nelisp-gpu-server-upload (photon-tensor-data b1)))
         (hw2  (nelisp-gpu-server-upload (photon-tensor-data w2)))
         (hb2  (nelisp-gpu-server-upload (photon-tensor-data b2)))
         (htgt (nelisp-gpu-server-upload (photon-tensor-data target)))
         (hlr  (nelisp-gpu-server-upload (vector lr)))
         (hin  (nelisp-gpu-server-upload (vector invn)))
         (slots (list (list 'res hx   (* seq in))    ;0  x
                      (list 'res hw1  (* hid in))    ;1  W1   (in place)
                      (list 'res hb1  hid)           ;2  b1   (in place)
                      (list 'res hw2  (* out hid))   ;3  W2   (in place)
                      (list 'res hb2  out)           ;4  b2   (in place)
                      (list 'res htgt (* seq out))   ;5  target
                      (list 'res hlr  1)             ;6  lr
                      (list 'res hin  1)             ;7  1/N
                      (cons 'tmp (* seq hid))        ;8  h1
                      (cons 'tmp (* seq hid))        ;9  a = gelu(h1)
                      (cons 'out (* seq out))        ;10 y (returned for loss)
                      (cons 'tmp (* seq out))        ;11 dy
                      (cons 'tmp (* out seq))        ;12 dy^T
                      (cons 'tmp (* out hid))        ;13 dW2
                      (cons 'tmp out)                ;14 db2
                      (cons 'tmp (* seq hid))        ;15 da
                      (cons 'tmp (* seq hid))        ;16 dh1
                      (cons 'tmp (* hid seq))        ;17 dh1^T
                      (cons 'tmp (* hid in))         ;18 dW1
                      (cons 'tmp hid)))              ;19 db1
         (disps (list (list 'linear    '(0 1 2 8)  (list seq in hid)  (funcall g (* seq hid)))
                      (list 'gelu      '(8 9)      (list (* seq hid)) (funcall g (* seq hid)))
                      (list 'linear    '(9 3 4 10) (list seq hid out) (funcall g (* seq out)))
                      (list 'sub-scale '(10 5 7 11)(list (* seq out)) (funcall g (* seq out)))
                      (list 'transpose '(11 12)    (list seq out)     (funcall g (* seq out)))
                      (list 'matmul    '(12 9 13)  (list out seq hid) (funcall g (* out hid)))
                      (list 'colsum    '(11 14)    (list seq out)     (funcall g out))
                      (list 'matmul    '(11 3 15)  (list seq out hid) (funcall g (* seq hid)))
                      (list 'gelu-bwd  '(15 8 16)  (list (* seq hid)) (funcall g (* seq hid)))
                      (list 'transpose '(16 17)    (list seq hid)     (funcall g (* seq hid)))
                      (list 'matmul    '(17 0 18)  (list hid seq in)  (funcall g (* hid in)))
                      (list 'colsum    '(16 19)    (list seq hid)     (funcall g hid))
                      (list 'sgd       '(1 18 6)   (list (* hid in))  (funcall g (* hid in)))
                      (list 'sgd       '(2 19 6)   (list hid)         (funcall g hid))
                      (list 'sgd       '(3 13 6)   (list (* out hid)) (funcall g (* out hid)))
                      (list 'sgd       '(4 14 6)   (list out)         (funcall g out))))
         (td (photon-tensor-data target)) (losses nil) (s 0))
    (while (< s steps)
      (let* ((y (car (nelisp-gpu-server-batch slots disps)))
             (acc 0.0) (i 0))
        (while (< i ntot)
          (let ((d (- (aref y i) (aref td i)))) (setq acc (+ acc (* d d))))
          (setq i (1+ i)))
        (push (* 0.5 invn acc) losses))
      (setq s (1+ s)))
    ;; read the trained resident weights back into the host tensors
    (cl-flet ((pull (handle n dst)
                (let ((v (car (nelisp-gpu-server-run2
                               'scale (list (list 'res handle n) (cons 'in (vector 1.0))
                                            (cons 'out n))
                               (list n) (funcall g n))))
                      (i 0))
                  (while (< i n) (aset dst i (aref v i)) (setq i (1+ i))))))
      (pull hw1 (* hid in)  (photon-tensor-data w1))
      (pull hb1 hid         (photon-tensor-data b1))
      (pull hw2 (* out hid) (photon-tensor-data w2))
      (pull hb2 out         (photon-tensor-data b2)))
    (dolist (h (list hx hw1 hb1 hw2 hb2 htgt hlr hin))
      (ignore-errors (nelisp-gpu-server-free h)))
    (nreverse losses)))

(provide 'nl-llm-gpu-train)
;;; nl-llm-gpu-train.el ends here
