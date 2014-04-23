#ifndef FIXPOINT_HPP
#define FIXPOINT_HPP
/**
 * @file /home/ryan/programming/nnet/fixpoint.hpp
 * @author Ryan Domigan <ryan_domigan@sutdents@uml.edu>
 * Created on Apr 21, 2014
 *
 * Relitively quick fixed-point implementnation, based on notes at
 * http://www.codeproject.com/Articles/37636/Fixed-Point-Class
 * and
 * http://www.drdobbs.com/cpp/fixed-point-arithmetic-types-for-c/184401992?pgno=2
 *
 * It looks like the CBC has onboard floating-point operations.
 * I'm going to hold off on this implementation for now.
 */
#include <type_traits>

template<class Int, size_t IntPart>
class FixedPoint {
  static_assert(std::is_integral<Int>::value
		, "FixedPoint overlays an integral type");
  Int value;
  
  static const Int exponent_mask = (-1) >> IntPart;
  static const Int fractional_mask = ~exponent_mask;

public:
  FixedPoint& operator+(const FixedPoint& in) {
    Int fraction = 
  }
};


#endif
