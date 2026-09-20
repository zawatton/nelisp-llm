;;; nl-llm-hadamard.el --- the rotation Bonsai folds into its weights  -*- lexical-binding: t; -*-

;; Ternary Bonsai applies an orthogonal rotation to each weight's input axis
;; before quantizing -- incoherence processing, which spreads outliers so that
;; a ternary grid loses less -- and folds it into the stored weights.  A runtime
;; either applies the matching transform to activations or gets wrong answers,
;; which is why the file declares it rather than assuming.  From the GGUF
;; metadata of Ternary-Bonsai-2-27B:
;;
;;     prism.hadamard.transform   normalized-sylvester-walsh-hadamard
;;     prism.hadamard.block_size  1024
;;     prism.hadamard.axis        input-last-dimension
;;     prism.hadamard.sign_mode   explicit
;;     prism.hadamard.sign_widths [5120, 6144, 17408]
;;     prism.hadamard.sign_values 28672 of them, one run per width
;;
;; All three widths are multiples of 1024, so the rotation is block diagonal
;; with no ragged tail.
;;
;; The Sylvester-Walsh matrix has H[i][j] = (-1)^popcount(i & j), is symmetric,
;; and normalised by 1/sqrt(n) is its own inverse.  That is worth knowing
;; because it means the forward and the inverse differ only in where the signs
;; go, and a test can assert the round trip is the identity.

;;; Code:

(defvar nl-llm-had-block 1024
  "The block size in force.  A power of two, as the transform needs.

A variable and not a constant because the size travels in the weight file's
`:hadamard-block' and a runtime that hardcodes 1024 answers wrongly for any
model that declares something else -- silently, since a block-diagonal
rotation of the wrong block size is still orthogonal and still preserves every
norm.  `nl-llm-bonsai-open' binds it from the header.")

(defun nl-llm-had--fwht (v base n)
  "In-place fast Walsh-Hadamard transform of N elements of V at BASE.
Unnormalised: the caller scales.  N must be a power of two."
  (let ((len 1))
    (while (< len n)
      (let ((i 0))
        (while (< i n)
          (let ((j i) (stop (+ i len)))
            (while (< j stop)
              (let* ((a (aref v (+ base j)))
                     (b (aref v (+ base j len))))
                (aset v (+ base j) (+ a b))
                (aset v (+ base j len) (- a b)))
              (setq j (1+ j))))
          (setq i (+ i len len))))
      (setq len (* 2 len)))
    v))

;;;###autoload
(defun nl-llm-had-rotate (x n signs &optional invert)
  "Rotate the N-long X in place by the model's transform; return X.
SIGNS is the N-long run of +/-1 for this width.  The forward applies the signs
and then the block Hadamard; INVERT reverses the order, which is the inverse
because the normalised Hadamard is an involution and a sign flip is its own
inverse.

N must be a whole number of blocks; the model's three widths all are."
  (unless (zerop (% n nl-llm-had-block))
    (error "nl-llm-had-rotate: %d is not a whole number of %d-blocks"
           n nl-llm-had-block))
  (let ((scale (/ 1.0 (sqrt (float nl-llm-had-block))))
        (nb (/ n nl-llm-had-block)))
    (if invert
        (progn
          (dotimes (b nb)
            (nl-llm-had--fwht x (* b nl-llm-had-block) nl-llm-had-block))
          (dotimes (i n) (aset x i (* (aref x i) scale (aref signs i)))))
      (dotimes (i n) (aset x i (* (aref x i) (aref signs i))))
      (dotimes (b nb)
        (nl-llm-had--fwht x (* b nl-llm-had-block) nl-llm-had-block))
      (dotimes (i n) (aset x i (* (aref x i) scale))))
    x))

;;;###autoload
(defun nl-llm-had-signs (values widths width)
  "The run of signs in VALUES for an input of WIDTH, given the WIDTHS list.
The file stores one run per distinct input width, concatenated."
  (let ((off 0) (found nil))
    (catch 'done
      (dolist (w widths)
        (when (= w width)
          (setq found (make-vector width 0.0))
          (dotimes (i width) (aset found i (float (aref values (+ off i)))))
          (throw 'done found))
        (setq off (+ off w))))
    (or found
        (error "nl-llm-had-signs: no sign run for width %d (have %S)"
               width widths))))

(provide 'nl-llm-hadamard)
;;; nl-llm-hadamard.el ends here
