;;; nl-llm-evolve.el --- transactional model evolution  -*- lexical-binding: t; -*-

;; Self-improvement must not train the only live model in place.  This module
;; gives an evolving agent a small champion/challenger boundary:
;;
;;   champion -> isolated copy -> propose/train -> evaluate -> promote or drop
;;
;; A failed experiment, a regression, or a gain below the configured gate leaves
;; the champion untouched.  Every attempt becomes a compact history entry.  The
;; proposal and training functions are deliberately model-agnostic so later
;; phases can mutate weights, attach experts, or rebuild an architecture behind
;; the same promotion rule.

;;; Code:

(require 'cl-lib)
(require 'photon-tensor)
(require 'photon-autograd)

(cl-defstruct (nl-llm-evolution
               (:constructor nl-llm-evolution--make))
  champion
  champion-score
  clone-fn
  evaluate-fn
  publish-fn
  min-delta
  generation
  attempts
  history
  promotion-gate-fn)

;;;###autoload
(defun nl-llm-evolve-copy-model (model)
  "Return an isolated copy of MODEL suitable for a candidate generation.

Autograd parameter values are copied into fresh leaf `pav' objects with zeroed
gradients.  If MODEL intentionally refers to one parameter more than once, that
weight tying is preserved inside the copy.  Lists, strings, and plain vectors
are copied recursively; symbols, numbers, and functions are shared."
  (let ((seen nil))
    (cl-labels
        ((copy-one
          (object)
          (cond
           ((pav-p object)
            (let ((known (assq object seen)))
              (if known
                  (cdr known)
                (let* ((value (pav-value object))
                       (value-copy
                        (photon-tensor
                         (copy-sequence (photon-tensor-shape value))
                         (copy-sequence (photon-tensor-data value))))
                       (result (photon-autograd-const value-copy)))
                  (push (cons object result) seen)
                  result))))
           ((consp object)
            (cons (copy-one (car object))
                  (copy-one (cdr object))))
           ((stringp object)
            (copy-sequence object))
           ((vectorp object)
            (let ((known (assq object seen)))
              (if known
                  (cdr known)
                (let* ((n (length object))
                       (result (make-vector n nil))
                       (i 0))
                  ;; Register before recursion so shared and cyclic vectors do
                  ;; not split into independent objects.
                  (push (cons object result) seen)
                  (while (< i n)
                    (aset result i (copy-one (aref object i)))
                    (setq i (1+ i)))
                  result))))
           (t object))))
      (copy-one model))))

(defun nl-llm-evolution--finite-number-p (value)
  "Return non-nil when VALUE is a finite real number."
  (and (numberp value)
       (= value value)
       (or (integerp value)
           (and (floatp value) (< (abs value) 1.7976931348623157e+308)))))

(defun nl-llm-evolution--score (evaluate model where)
  "Evaluate MODEL with EVALUATE and return a valid numeric score for WHERE."
  (let ((score (funcall evaluate model)))
    (unless (nl-llm-evolution--finite-number-p score)
      (error "%s: evaluator returned invalid score %S" where score))
    score))

(defun nl-llm-evolution--record (state entry)
  "Append ENTRY to STATE's newest-first audit history and return ENTRY."
  (setf (nl-llm-evolution-history state)
        (cons entry (nl-llm-evolution-history state)))
  entry)

(defun nl-llm-evolution--accept-scored
    (state candidate score expected-generation expected-score metadata)
  "Accept a trusted scored CANDIDATE for STATE.

EXPECTED-GENERATION and EXPECTED-SCORE are an optimistic concurrency boundary:
they must still describe STATE's champion before any mutation.  CANDIDATE is
cloned before it can become the champion.  This helper is shared by the
synchronous and asynchronous completion paths; SCORE is deliberately supplied
by the trusted host rather than evaluated here."
  (unless (nl-llm-evolution-p state)
    (error "nl-llm-evolution-accept-scored: STATE is not an evolution state"))
  (unless (and (integerp expected-generation)
               (= expected-generation (nl-llm-evolution-generation state)))
    (error "stale evolution parent generation: expected %S, current %S"
           expected-generation (nl-llm-evolution-generation state)))
  (unless (and (nl-llm-evolution--finite-number-p expected-score)
               (= expected-score (nl-llm-evolution-champion-score state)))
    (error "stale evolution parent score: expected %S, current %S"
           expected-score (nl-llm-evolution-champion-score state)))
  (unless (nl-llm-evolution--finite-number-p score)
    (error "nl-llm-evolution-accept-scored: invalid score %S" score))
  (let* ((attempt (1+ (nl-llm-evolution-attempts state)))
         (generation expected-generation)
         (before expected-score)
         (stage 'clone))
    (setf (nl-llm-evolution-attempts state) attempt)
    (condition-case err
        (let* ((isolated
                (funcall (nl-llm-evolution-clone-fn state) candidate))
               (delta (- score before))
               (promote
                (> score (+ before (nl-llm-evolution-min-delta state))))
               (entry
                (list :attempt attempt
                      :status (if promote 'promoted 'rejected)
                      :generation-before generation
                      :generation-after (if promote (1+ generation) generation)
                      :score-before before :score-after score :delta delta
                      :metadata metadata)))
          (if (not promote)
              (nl-llm-evolution--record state entry)
            (progn
              (when (nl-llm-evolution-promotion-gate-fn state)
                (setq stage 'promotion-gate)
                ;; Both arguments are detached.  Gate mutations therefore
                ;; cannot alter the live champion or accepted candidate.
                (let ((gate-result
                       (funcall
                        (nl-llm-evolution-promotion-gate-fn state)
                        (funcall (nl-llm-evolution-clone-fn state)
                                 (nl-llm-evolution-champion state))
                        (funcall (nl-llm-evolution-clone-fn state) isolated))))
                  (unless (or (eq gate-result t) (null gate-result))
                    (error "promotion gate must return exactly t or nil"))
                  (if gate-result
                      (setq entry
                            (append entry (list :promotion-gate 'passed)))
                    (setq promote nil
                          entry (plist-put entry :status 'rejected))
                    (setq entry (plist-put entry :generation-after generation))
                    (setq entry
                          (append entry (list :promotion-gate 'rejected)))))
                ;; A trusted gate may run a nested evolution operation.  Do
                ;; not publish an outer result against a changed parent.
                (unless (and (= generation
                                (nl-llm-evolution-generation state))
                             (= before
                                (nl-llm-evolution-champion-score state)))
                  (error "stale evolution state after promotion gate")))
              (if (not promote)
                  (nl-llm-evolution--record state entry)
                (setq stage 'publish)
                (when (nl-llm-evolution-publish-fn state)
                  (let ((publication
                         (funcall (nl-llm-evolution-publish-fn state)
                                  isolated (copy-tree entry) state)))
                    (when publication
                      (setq entry
                            (append entry
                                    (list :publication
                                          (copy-tree publication)))))))
                (setq stage 'commit)
                (setf (nl-llm-evolution-champion state) isolated
                      (nl-llm-evolution-champion-score state) score
                      (nl-llm-evolution-generation state) (1+ generation))
                (nl-llm-evolution--record state entry)))))
      (error
       (nl-llm-evolution--record
        state (list :attempt attempt :status 'error
                    :generation-before generation
                    :generation-after
                    (if (eq stage 'promotion-gate)
                        (nl-llm-evolution-generation state)
                      generation)
                    :score-before before :score-after nil :delta nil
                    :stage stage :error (format "%S" err)
                    :metadata metadata))))))

;;;###autoload
(defun nl-llm-evolution-accept-scored
    (state candidate score expected-generation expected-score &optional metadata)
  "Accept a trusted asynchronous completion into STATE.

The parent generation and score must match exactly.  The candidate is detached
by STATE's clone function, and promotion uses the same strict gate and
publish-before-commit transaction as `nl-llm-evolution-step'."
  (nl-llm-evolution--accept-scored
   state candidate score expected-generation expected-score metadata))

;;;###autoload
(cl-defun nl-llm-evolution-new
    (model evaluate &key (clone #'nl-llm-evolve-copy-model) publish
           promotion-gate (min-delta 0.0) (generation 0))
  "Create a transactional evolution state around MODEL.

EVALUATE is a deterministic function from model to a numeric score where larger
is better.  CLONE must return an isolated candidate; the default understands the
`pav'-based models built by `nl-llm-agent-improve-model'.  MIN-DELTA is the
strict minimum score gain required for promotion.  The initial champion is also
cloned, so later evolution cannot mutate the caller's parent model.  PUBLISH,
when non-nil, receives (CANDIDATE PROMOTION-ENTRY STATE) after the gate passes
but before live champion state changes; a publish failure cancels promotion.
GENERATION restores the number of an already-published initial champion."
  (unless (functionp evaluate)
    (error "nl-llm-evolution-new: EVALUATE must be a function"))
  (unless (functionp clone)
    (error "nl-llm-evolution-new: CLONE must be a function"))
  (when (and publish (not (functionp publish)))
    (error "nl-llm-evolution-new: PUBLISH must be nil or a function"))
  (when (and promotion-gate (not (functionp promotion-gate)))
    (error "nl-llm-evolution-new: PROMOTION-GATE must be nil or a function"))
  (unless (and (numberp min-delta) (>= min-delta 0.0))
    (error "nl-llm-evolution-new: MIN-DELTA must be non-negative, got %S"
           min-delta))
  (unless (and (integerp generation) (>= generation 0))
    (error "nl-llm-evolution-new: GENERATION must be non-negative, got %S"
           generation))
  (let* ((champion (funcall clone model))
         (score (nl-llm-evolution--score
                 evaluate champion "nl-llm-evolution-new")))
    (nl-llm-evolution--make
     :champion champion
     :champion-score score
     :clone-fn clone
     :evaluate-fn evaluate
     :publish-fn publish
     :promotion-gate-fn promotion-gate
     :min-delta min-delta
     :generation generation
     :attempts 0
     :history nil)))

;;;###autoload
(cl-defun nl-llm-evolution-step (state propose &key train metadata)
  "Run one isolated challenger experiment in evolution STATE.

PROPOSE receives (CANDIDATE STATE) and mutates CANDIDATE in place.  Its return
value is deliberately ignored: Lisp mutators such as `aset' return the assigned
value, which must never be mistaken for a replacement model.  Optional TRAIN
receives the resulting (CANDIDATE STATE) and may train it in place.  The
candidate is evaluated only after both hooks finish.

The candidate replaces the champion only when its score is strictly greater
than CHAMPION-SCORE + MIN-DELTA.  Regressions, insufficient gains, and errors
are contained.  Return an audit plist with :status `promoted', `rejected', or
`error'.  METADATA is copied into that entry without interpretation."
  (unless (nl-llm-evolution-p state)
    (error "nl-llm-evolution-step: STATE is not an evolution state"))
  (unless (functionp propose)
    (error "nl-llm-evolution-step: PROPOSE must be a function"))
  (when (and train (not (functionp train)))
    (error "nl-llm-evolution-step: TRAIN must be nil or a function"))
  (let* ((state-generation (nl-llm-evolution-generation state))
         (state-score (nl-llm-evolution-champion-score state))
         (candidate nil)
         (stage 'clone))
    (condition-case err
        (progn
          (setq candidate
                (funcall (nl-llm-evolution-clone-fn state)
                         (nl-llm-evolution-champion state)))
          (setq stage 'propose)
          (funcall propose candidate state)
          (when train
            (setq stage 'train)
            (funcall train candidate state))
          (setq stage 'evaluate)
          (nl-llm-evolution--accept-scored
           state candidate
           (nl-llm-evolution--score (nl-llm-evolution-evaluate-fn state)
                                    candidate "nl-llm-evolution-step")
           state-generation state-score metadata))
      (error
       ;; Preserve the historical containment behavior for synchronous hooks.
       (let ((attempt (1+ (nl-llm-evolution-attempts state))))
         (setf (nl-llm-evolution-attempts state) attempt)
         (nl-llm-evolution--record
          state (list :attempt attempt :status 'error
                      :generation-before state-generation
                      :generation-after state-generation
                      :score-before state-score :score-after nil :delta nil
                      :stage stage :error (format "%S" err)
                      :metadata metadata)))))))

(provide 'nl-llm-evolve)
;;; nl-llm-evolve.el ends here
