;;; lora-demo.el --- adapt the imported model, and check it stayed a model -*- lexical-binding: t -*-

;; Everything up to here has verified the machinery: gradients against finite
;; differences, the GPU against the CPU, a step against a reference.  None of
;; it shows the thing the document is about -- that an imported model can be
;; *adapted*.  This runs the loop on the real Qwen3-0.6B and asks three
;; questions, of which only the first is usually asked:
;;
;;   1. does the loss descend?
;;   2. does the model's prediction on a trained prompt actually change?
;;   3. does an untrained prompt stay where it was?
;;
;; The third is the one that makes the other two mean anything.  A LoRA that
;; has simply broken the model also drives the training loss down and also
;; changes the trained prediction; the control is what separates adaptation
;; from damage.
;;
;; The facts taught are invented, and deliberately so: the base model cannot
;; know them, so any movement toward them is attributable to the adapter
;; rather than to something it already contained.
;;
;; Trained and evaluated through the same W8A8 path, which is train/test match
;; for a model served by `nl-llm-wgpu-next-token'.  See the "One whole step"
;; section of docs/design/08-weight-import.org for why that is a choice.

;; One structural note, because it changes what the result means: the loop
;; passes *one* LORAS plist to every block, so a rank-8 adapter here is tied
;; across all 28 layers and its gradient is the sum over them, rather than the
;; usual per-layer adapter.  That is what the API does and it is a real
;; parameterization; it is simply not the one the word "rank 8" usually
;; denotes, so the parameter count is 8 x (2048 + 1024) + 8 x (1024 + 1024)
;; once, not twenty-eight times.

(require 'nl-llm-weights-gpu)
(require 'nl-llm-weights-train)
(require 'nl-llm-qwen-tokenizer)
(require 'nl-llm-lora)

(defconst lora-demo-examples
  '((:prompt "The build server is called" :completion " Granite")
    (:prompt "The project codename is" :completion " Bluefin")
    (:prompt "The release train leaves on" :completion " Thursday"))
  "Invented facts, so movement toward them cannot come from the base model.")

(defconst lora-demo-control
  '(:prompt "The capital of France is" :completion " Paris")
  "Untrained, and one the base model is known to get right: the Phase 2c
milestone scored token 12095 here.  If this moves, the adapter is not adapting.")

(defun lora-demo--probe (layers head cfg loras tok example)
  "Score EXAMPLE's first completion token after its prompt.
Returns (TOP-TEXT TARGET-TEXT TARGET-PROB TARGET-RANK).

The *first* token, which is not always the whole completion: \=" Bluefin\="
is two tokens, \=" Blue\=" and \="fin\=", so what is scored here is
\=" Blue\=".  Reporting the completion string next to a rank measured on its
first token reads as a contradiction -- top \=" Blue\=" and \=" Bluefin\="
at rank 1 at the same time -- so this returns the target's own text and the
caller prints that."
  (let* ((dim (plist-get cfg :dim))
         (ids (nl-llm-qwen-tok-encode tok (plist-get example :prompt)))
         (target (car (nl-llm-qwen-tok-encode tok (plist-get example :completion))))
         (fw (nl-llm-wtrain-forward layers head ids cfg loras))
         (hidden (nth 0 fw))
         (logits (nl-llm-wb--apply (plist-get head :lin) hidden (* (1- (length ids)) dim)))
         (vocab (length logits))
         (mx -1.0e30) (sum 0.0) (top 0) (rank 1))
    (dotimes (j vocab) (when (> (aref logits j) mx) (setq mx (aref logits j) top j)))
    (dotimes (j vocab) (setq sum (+ sum (exp (- (aref logits j) mx)))))
    (dotimes (j vocab) (when (> (aref logits j) (aref logits target))
                         (setq rank (1+ rank))))
    (list (condition-case nil (nl-llm-qwen-tok-decode tok (list top)) (error "?"))
          (condition-case nil (nl-llm-qwen-tok-decode tok (list target)) (error "?"))
          (/ (exp (- (aref logits target) mx)) sum)
          rank)))

(defun lora-demo--report (tag layers head cfg loras tok)
  (dolist (ex (append lora-demo-examples (list lora-demo-control)))
    (let ((p (lora-demo--probe layers head cfg loras tok ex)))
      (message "%-6s %-32s top %-12S target %-10S p %.3e rank %d"
               tag (plist-get ex :prompt) (nth 0 p) (nth 1 p) (nth 2 p) (nth 3 p)))))

(let* ((wts (nl-llm-weights-open "build/donor/qwen3-0.6b/weights.bin"))
       (tok (nl-llm-qwen-tok-load "build/donor/qwen3-0.6b/tokenizer.bin"))
       (steps (string-to-number (or (getenv "LORA_DEMO_STEPS") "36")))
       (lr (string-to-number (or (getenv "LORA_DEMO_LR") "0.02"))))
  (nelisp-gpu-server-start)
  (let* ((t0 (float-time))
         (sess (nl-llm-wgpu-open-model wts))
         (layers (plist-get sess :layers))
         (head (plist-get sess :head))
         (tbl (plist-get sess :table))
         (cfg (plist-get sess :cfg))
         (lay0 (car layers))
         (loras (list :wq (let ((l (nl-llm-wf-layer-lin lay0 :wq)))
                            (nl-llm-lora-make (nl-llm-weights-lin-rows l)
                                              (nl-llm-weights-lin-cols l) 8 16.0 1))
                      :wv (let ((l (nl-llm-wf-layer-lin lay0 :wv)))
                            (nl-llm-lora-make (nl-llm-weights-lin-rows l)
                                              (nl-llm-weights-lin-cols l) 8 16.0 2))))
         (examples (vconcat lora-demo-examples)))
    (message "RESIDENT %.0fs  %d buffers  steps %d  lr %s  adapters :wq :wv rank 8"
             (- (float-time) t0) (hash-table-count tbl) steps lr)
    (nl-llm-wgpu-with-linears tbl
      (lora-demo--report "BEFORE" layers head cfg loras tok)
      (let ((losses (nl-llm-wtrain-fit
                     layers head examples tok cfg loras steps lr
                     (lambda (step loss len)
                       (message "STEP %3d  loss %.4f  (%d tokens)" step loss len)))))
        (message "LOSS first %.4f  last %.4f" (car losses) (car (last losses))))
      (lora-demo--report "AFTER" layers head cfg loras tok))
    (nl-llm-wgpu-close-model sess))
  (nelisp-gpu-server-stop)
  (message "DONE"))
