#ifndef AGGREGATION_HPP
#define AGGREGATION_HPP
/**
 * @file /home/ryan/programming/nnet/aggregation.hpp
 * @author Ryan Domigan <ryan_domigan@sutdents@uml.edu>
 * Created on Apr 10, 2014
 *
 * Aggregation for NNet and std::array
 */
#include <array>
#include <tuple>

template<class FN, class T, size_t n>
void ref_map(const FN fn
	     , std::array<T,n> &dst
	     , const std::array<T,n> &aux) {
  typedef std::array<T,n> Array;
  for(size_t i = 0; i < std::tuple_size< Array >::value; ++i) {
    dst[i] = fn(dst[i], aux[i]);
  }
}

template<class FN, class T, size_t n>
void fold(const FN fn
	  , const std::array<T,n> &src) {
  typedef std::array<T,n> Array;
  for(size_t i = 0; i < std::tuple_size< Array >::value; ++i)
    fn(src[i]);
}




#endif
