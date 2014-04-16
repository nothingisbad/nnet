#ifndef AGGREGATION_HPP
#define AGGREGATION_HPP
/**
 * @file /home/ryan/programming/nnet/aggregation.hpp
 * @author Ryan Domigan <ryan_domigan@sutdents@uml.edu>
 * Created on Apr 10, 2014
 *
 * Aggregation for NNet and std::array.
 * I haven't come up with a good way to check the input types for dimention matches.  The compiler will still throw a fit
 * in some cases, but it would be nice to have some static_asserts.
 */
#include <array>
#include <tuple>
#include <algorithm>
#include <iostream>

template<int Start = 0, int End = 0>
struct Map {
  template<class FN, class T, class ... Aux>
  static void apply(FN fn , T& dst, Aux& ... aux) {
    const static size_t last =
      End <= 0 ? std::min( {std::tuple_size< T >::value, std::tuple_size<Aux>::value ...} ) + End
               : End;

    for(size_t i = Start; i < last ; ++i)
      fn(dst[i], aux[i]...);
  }
};

template<class FN, class T, class ... Aux>
void ref_map(FN fn , T& dst, Aux& ... aux) { Map<>::apply(fn,dst,aux...); }

template<class FN, class T>
void array_fold(FN fn, T src) {
  for(size_t i = 0; i < std::tuple_size< T >::value; ++i)
    fn(src[i]);
}

template<class A>
std::ostream& print_array(const A& aa, std::ostream& out) {
  using namespace std;
  if(aa.size() == 0)
    return cout << "[]";

  cout << "[" << aa[0];
  Map<1>::apply([&](float ff) { out << ", " << ff; }, aa);
  return cout << "]";
}

template<class A>
std::ostream& print_array(const A& aa) { return print_array(aa, std::cout); }

#endif
