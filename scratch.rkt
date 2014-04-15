#lang racket

(define (sigmoid x)
  (/ 1 (+ 1 (exp (- x)))))

(define (grad-sigmoid x)
  (let ((sig (sigmoid x)))
    (* sig (- 1 sig))))

(define (delta-theta theta-out delta-out activation-input)
  (* theta-out delta-out (grad-sigmoid activation-input)) )

(define (delta-theta-given-sm theta-out delta-out activation)
  (* theta-out delta-out (* activation (- 1 activation))) )
