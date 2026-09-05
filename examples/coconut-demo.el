;;; coconut-demo.el --- run the staged continuous-thought curriculum  -*- lexical-binding: t; -*-

;; This CPU demonstration trains a tiny two-block Coconut model on four-digit
;; chain addition.  At each curriculum boundary it reports the epoch loss
;; trajectory, held-out final-answer accuracy in the corresponding latent
;; mode, and emitted language-token count against the stage-zero CoT budget.

(add-to-list 'load-path (expand-file-name "lisp"))
(add-to-list 'load-path (expand-file-name "../nelisp-photon/lisp"))
(require 'photon-tensor)
(require 'photon-autograd)
(require 'nl-llm-autograd)
(require 'nl-llm-coconut)

(random "fixed-seed")

(defun coconut-demo--evaluate (model data stage c)
  "Return (ANSWER-ACCURACY MEAN-LANGUAGE-TOKENS) for MODEL on DATA.
STAGE and C select the matching curriculum's latent-thought count."
  (let ((correct 0)
        (token-count 0)
        (count 0))
    (dolist (item data)
      (let* ((example (nl-llm-coconut-stage-example item stage c))
             (generated (nl-llm-coconut-generate
                         model (car example) (* stage c)
                         (length (cdr example))))
             (answer (and generated (car (last generated)))))
        (when (and answer (= answer (plist-get item :a)))
          (setq correct (1+ correct)))
        (setq token-count (+ token-count (length generated)))
        (setq count (1+ count))))
    (list (/ (float correct) count)
          (/ (float token-count) count))))

(defun coconut-demo--trajectory-string (trajectory)
  "Format an epoch-loss TRAJECTORY for the demonstration table."
  (mapconcat (lambda (value) (format "%.4f" value)) trajectory " -> "))

(let* ((k 4)
       (c 1)
       (stages 2)
       (epochs 3)
       (cot-token-budget k)
       (train-data (nl-llm-coconut-task-chain-add k 64 2026))
       (held-out (nl-llm-coconut-task-chain-add k 32 3030))
       (model (nl-llm-coconut-model-new
               :vocab 12 :dim 16 :heads 2 :kv-heads 1 :ff 16
               :nblocks 2 :seed 53))
       (stage-metrics nil)
       (trajectories
        (nl-llm-coconut-train
         model train-data :stages stages :epochs epochs :lr 0.3 :c c
         :trace
         (lambda (stage epoch _mean-loss)
           (when (= epoch (1- epochs))
             (push (cons stage
                         (coconut-demo--evaluate model held-out stage c))
                   stage-metrics)))))
       (pass t)
       (stage 0))
  (setq stage-metrics (nreverse stage-metrics))
  (princ "Coconut chain-add curriculum (K=4, train=64, held-out=32, dim=16, blocks=2, c=1, SGD lr=0.3)\n")
  (princ "\n")
  (princ "| stage | thoughts | epoch mean loss                 | held-out answer | language tokens / stage-0 CoT |\n")
  (princ "|-------+----------+---------------------------------+-----------------+-------------------------------|\n")
  (dolist (trajectory trajectories)
    (let* ((metrics (cdr (assq stage stage-metrics)))
           (accuracy (car metrics))
           (mean-tokens (nth 1 metrics)))
      (unless (< (car (last trajectory)) (car trajectory))
        (setq pass nil))
      (princ (format "| %5d | %8d | %-31s | %14.1f%% | %10.2f / %-15d |\n"
                     stage (* stage c)
                     (coconut-demo--trajectory-string trajectory)
                     (* 100.0 accuracy) mean-tokens cot-token-budget)))
    (setq stage (1+ stage)))
  (princ "\nContinuous thoughts are hidden-state feedback rows, not emitted language tokens.\n")
  (princ (format "COCONUT-DEMO=%s\n" (if pass "PASS" "FAIL")))
  (kill-emacs (if pass 0 1)))
;;; coconut-demo.el ends here
