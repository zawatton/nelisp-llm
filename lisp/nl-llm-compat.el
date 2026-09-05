;;; nl-llm-compat.el --- NeLisp standalone compatibility shims  -*- lexical-binding: t; -*-

;; This module supplies small Emacs-compatible definitions that the standalone
;; NeLisp reader does not yet provide.  Every shim is guarded so a native
;; implementation always wins.  The trigonometric kernels use Cody-Waite range
;; reduction and fdlibm minimax polynomials; no foreign-function interface is
;; required.

;;; Code:

(declare-function nl-llm-compat--reduce-pio2 "nl-llm-compat")
(declare-function nl-llm-compat--sin-kernel "nl-llm-compat")
(declare-function nl-llm-compat--cos-kernel "nl-llm-compat")
(declare-function nl-llm-compat--atan-kernel "nl-llm-compat")

(unless (and (fboundp 'sin)
             (fboundp 'cos)
             (fboundp 'tan)
             (fboundp 'atan))
  (unless (fboundp 'nl-llm-compat--reduce-pio2)
    (defun nl-llm-compat--reduce-pio2 (x)
      "Return (QUADRANT . REMAINDER) after reducing X by pi/2.
REMAINDER lies in [-pi/4, pi/4].  The split pi/2 constants retain precision
when X is much larger than one."
      (let* ((xf (float x))
             (n (round (* xf 0.63661977236758134308)))
             (r (- (- (- xf (* n 1.57079632673412561417))
                         (* n 6.07710050630396597660e-11))
                    (* n 2.02226624879595063154e-21))))
        (cons (mod n 4) r))))

  (unless (fboundp 'nl-llm-compat--sin-kernel)
    (defun nl-llm-compat--sin-kernel (x)
      "Return sin(X) for X in [-pi/4, pi/4]."
      (let ((z (* x x)))
        (+ x
           (* x z
              (+ -1.66666666666666324348e-01
                 (* z
                    (+ 8.33333333332248946124e-03
                       (* z
                          (+ -1.98412698298579493134e-04
                             (* z
                                (+ 2.75573137070700676789e-06
                                   (* z
                                      (+ -2.50507602534068634195e-08
                                         (* z 1.58969099521155010221e-10)))))))))))))))

  (unless (fboundp 'nl-llm-compat--cos-kernel)
    (defun nl-llm-compat--cos-kernel (x)
      "Return cos(X) for X in [-pi/4, pi/4]."
      (let ((z (* x x)))
        (+ (- 1.0 (* 0.5 z))
           (* z z
              (+ 4.16666666666666019037e-02
                 (* z
                    (+ -1.38888888888741095749e-03
                       (* z
                          (+ 2.48015872894767294178e-05
                             (* z
                                (+ -2.75573143513906633035e-07
                                   (* z
                                      (+ 2.08757232129817482790e-09
                                         (* z -1.13596475577881948265e-11)))))))))))))))

  (unless (fboundp 'nl-llm-compat--atan-kernel)
    (defun nl-llm-compat--atan-kernel (x)
      "Return atan(X) using fdlibm argument reduction and polynomials."
      (let* ((negative (< x 0.0))
             (ax (if negative (- (float x)) (float x)))
             (id -1)
             (r ax))
        (cond
         ((< ax 0.4375))
         ((< ax 0.6875)
          (setq id 0 r (/ (- (* 2.0 ax) 1.0) (+ 2.0 ax))))
         ((< ax 1.1875)
          (setq id 1 r (/ (- ax 1.0) (+ ax 1.0))))
         ((< ax 2.4375)
          (setq id 2 r (/ (- ax 1.5) (+ 1.0 (* 1.5 ax)))))
         (t
          (setq id 3 r (/ -1.0 ax))))
        (let* ((z (* r r))
               (w (* z z))
               (even (+ 4.97687799461593236017e-02
                        (* w 1.62858201153657823623e-02)))
               (even (+ 6.66107313738753120669e-02 (* w even)))
               (even (+ 9.09088713343650656196e-02 (* w even)))
               (even (+ 1.42857142725034663711e-01 (* w even)))
               (s1 (* z (+ 3.33333333333329318027e-01 (* w even))))
               (odd (+ -5.83357013379057348645e-02
                       (* w -3.65315727442169191654e-02)))
               (odd (+ -7.69187620504482999495e-02 (* w odd)))
               (odd (+ -1.11111104054623557880e-01 (* w odd)))
               (s2 (* w (+ -1.99999999998764832476e-01 (* w odd))))
               (poly (+ s1 s2))
               (value
                (if (< id 0)
                    (- r (* r poly))
                  (let ((hi (cond ((= id 0) 4.63647609000806093515e-01)
                                  ((= id 1) 7.85398163397448278999e-01)
                                  ((= id 2) 9.82793723247329054082e-01)
                                  (t 1.57079632679489655800e+00)))
                        (lo (cond ((= id 0) 2.26987774529616870924e-17)
                                  ((= id 1) 3.06161699786838301793e-17)
                                  ((= id 2) 1.39033110312309984516e-17)
                                  (t 6.12323399573676603587e-17))))
                    (- hi (- (- (* r poly) lo) r))))))
          (if negative (- value) value)))))

  (unless (fboundp 'sin)
    (defun sin (x)
      "Return the sine of X, measured in radians."
      (let* ((reduced (nl-llm-compat--reduce-pio2 x))
             (quadrant (car reduced))
             (r (cdr reduced)))
        (cond ((= quadrant 0) (nl-llm-compat--sin-kernel r))
              ((= quadrant 1) (nl-llm-compat--cos-kernel r))
              ((= quadrant 2) (- (nl-llm-compat--sin-kernel r)))
              (t (- (nl-llm-compat--cos-kernel r)))))))

  (unless (fboundp 'cos)
    (defun cos (x)
      "Return the cosine of X, measured in radians."
      (let* ((reduced (nl-llm-compat--reduce-pio2 x))
             (quadrant (car reduced))
             (r (cdr reduced)))
        (cond ((= quadrant 0) (nl-llm-compat--cos-kernel r))
              ((= quadrant 1) (- (nl-llm-compat--sin-kernel r)))
              ((= quadrant 2) (- (nl-llm-compat--cos-kernel r)))
              (t (nl-llm-compat--sin-kernel r))))))

  (unless (fboundp 'tan)
    (defun tan (x)
      "Return the tangent of X, measured in radians."
      (/ (sin x) (cos x))))

  (unless (fboundp 'atan)
    (defun atan (y &optional x)
      "Return the inverse tangent of Y, or the angle of Y and X.
With X, return the angle in radians of the vector whose coordinates are X and
Y, using the signs of both arguments to select the quadrant."
      (if (null x)
          (nl-llm-compat--atan-kernel y)
        (let ((yf (float y))
              (xf (float x)))
          (cond ((> xf 0.0) (nl-llm-compat--atan-kernel (/ yf xf)))
                ((< xf 0.0)
                 (if (>= yf 0.0)
                     (+ (nl-llm-compat--atan-kernel (/ yf xf))
                        3.14159265358979323846)
                   (- (nl-llm-compat--atan-kernel (/ yf xf))
                      3.14159265358979323846)))
                ((> yf 0.0) 1.57079632679489661923)
                ((< yf 0.0) -1.57079632679489661923)
                (t 0.0)))))))

(unless (fboundp 'user-error)
  (defun user-error (format-string &rest args)
    "Signal an error using FORMAT-STRING and ARGS."
    (apply #'error format-string args)))

(unless (fboundp 'call-process-shell-command)
  (defun call-process-shell-command
      (command &optional infile destination display &rest args)
    "Run shell COMMAND synchronously through /bin/sh.
INFILE, DESTINATION, DISPLAY, and ARGS have the meanings used by
`call-process'."
    (apply #'call-process "/bin/sh" infile destination display
           "-c" command args)))

(unless (fboundp 'called-interactively-p)
  (defun called-interactively-p (&optional _kind)
    "Return non-nil when the containing function was called interactively.
The standalone NeLisp reader has no interactive command loop, so this fallback
always returns nil."
    nil))

(unless (fboundp 'check-parens)
  (defun check-parens ()
    "Signal `user-error' if the current buffer has unbalanced delimiters."
    (let* ((text (buffer-string))
           (length (length text))
           (index 0)
           (stack nil)
           (in-string nil)
           (escaped nil)
           (line-comment nil)
           (block-depth 0))
      (while (< index length)
        (let ((char (aref text index))
              (next (and (< (1+ index) length) (aref text (1+ index)))))
          (cond
           (line-comment
            (when (= char ?\n)
              (setq line-comment nil)))
           ((> block-depth 0)
            (cond ((and (= char ?#) (= next ?|))
                   (setq block-depth (1+ block-depth)
                         index (1+ index)))
                  ((and (= char ?|) (= next ?#))
                   (setq block-depth (1- block-depth)
                         index (1+ index)))))
           (in-string
            (cond (escaped (setq escaped nil))
                  ((= char ?\\) (setq escaped t))
                  ((= char ?\") (setq in-string nil))))
           ((= char ?\;) (setq line-comment t))
           ((and (= char ?#) (= next ?|))
            (setq block-depth 1 index (1+ index)))
           ((= char ?\") (setq in-string t))
           ((= char ??)
            (when (< (1+ index) length)
              (setq index (1+ index))
              (when (and (= (aref text index) ?\\)
                         (< (1+ index) length))
                (setq index (1+ index)))))
           ((= char ?\() (push ?\) stack))
           ((= char ?\[) (push ?\] stack))
           ((or (= char ?\)) (= char ?\]))
            (unless (and stack (= char (car stack)))
              (user-error "Unmatched bracket or quote"))
            (setq stack (cdr stack)))))
        (setq index (1+ index)))
      (when (or stack in-string (> block-depth 0))
        (user-error "Unmatched bracket or quote"))
      nil)))

(unless (fboundp 'file-relative-name)
  (defun file-relative-name (filename &optional directory)
    "Return FILENAME relative to DIRECTORY when FILENAME is below it."
    (let* ((file (expand-file-name filename))
           (dir (file-name-as-directory
                 (expand-file-name (or directory default-directory)))))
      (if (string-prefix-p dir file)
          (substring file (length dir))
        file))))

(unless (fboundp 'directory-files-recursively)
  (defun directory-files-recursively
      (directory regexp &optional include-directories _predicate _follow-symlinks)
    "Return files below DIRECTORY whose names match REGEXP.
When INCLUDE-DIRECTORIES is non-nil, include matching directories as well."
    (let ((entries (directory-files directory t nil t))
          (result nil))
      (dolist (entry entries)
        (let ((name (file-name-nondirectory entry)))
          (unless (or (equal name ".") (equal name ".."))
            (if (file-directory-p entry)
                (progn
                  (when (and include-directories (string-match-p regexp entry))
                    (push entry result))
                  (setq result
                        (append result
                                (directory-files-recursively entry regexp
                                                             include-directories))))
              (when (string-match-p regexp entry)
                (push entry result))))))
      result)))

;; NeLisp already supplies every subr-x function used by nelisp-llm, but does
;; not advertise the feature.  This guarded feature alias lets existing
;; requires succeed without shadowing any function implementation.
(unless (featurep 'subr-x)
  (provide 'subr-x))

(provide 'nl-llm-compat)
;;; nl-llm-compat.el ends here
