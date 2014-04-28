#ifndef NNET_UTILITY_HPP
#define NNET_UTILITY_HPP
/**
 * @file /home/ryan/programming/nnet/utility.hpp
 * @author Ryan Domigan <ryan_domigan@sutdents@uml.edu>
 * Created on Apr 10, 2014
 *
 * Utility for NNet and std::array.
 * I haven't come up with a good way to check the input types for dimention matches.  The compiler will still throw a fit
 * in some cases, but it would be nice to have some static_asserts.
 */
#include <array>
#include <tuple>
#include <algorithm>
#include <iostream>
#include <type_traits>

template<class Num>
Num sigmoid(const Num& input) {
  static_assert( std::is_floating_point<Num>::value
		 , "sigmoid only makes sense for floating point numbers");
  /* speed up following example at http://msdn.microsoft.com/en-us/magazine/jj658979.aspx */
  if(input < -45.0) return 0.0;
  if(input > 45.0) return 1.0;
  return 1.0 / (1.0 + exp(-input));
}

/**************************************/
/*     _                              */
/*    / \   _ __ _ __ __ _ _   _ ___  */
/*   / _ \ | '__| '__/ _` | | | / __| */
/*  / ___ \| |  | | | (_| | |_| \__ \ */
/* /_/   \_\_|  |_|  \__,_|\__, |___/ */
/*                         |___/      */
/**************************************/
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
void map_array(FN fn , T& dst, Aux& ... aux) { Map<>::apply(fn,dst,aux...); }

/* template<class FN, class T> */
/* void array_fold(FN fn, T src) { */
/*   for(size_t i = 0; i < std::tuple_size< T >::value; ++i) */
/*     fn(src[i]); */
/* } */

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

/************************/
/*  __  __ ____  _      */
/* |  \/  |  _ \| |     */
/* | |\/| | |_) | |     */
/* | |  | |  __/| |___  */
/* |_|  |_|_|   |_____| */
/************************/
template<size_t N>
struct intC : public std::integral_constant<size_t, N>::type {};


/* Well, std library has some metaprogramming stuff now, but it's missing a lot. */
struct _void {
  typedef _void type;
  /* eats constructor args so I can use it to terminate
     recursive constructors */
  template<class ... T>
  _void(T...) {}
};

template<class T> struct Identity { typedef T type; };

template<class T, class U = typename T::type>
struct IsVoid : std::false_type {};

template<class T>
struct IsVoid<T,_void> : std::true_type {};

template<size_t ... Rest> struct Nums;

namespace detail {
  template<class T> struct GetDepth : public intC<T::depth + 1>::type {};

  template<> struct GetDepth<_void> : public intC<0>::type {};
}

template<size_t N, class NNs>
struct ConsNums {
  typedef NNs Next;
  static const size_t value = N;
  static const size_t depth = detail::GetDepth<Next>::value;

  typedef ConsNums<N, NNs> type;
};


template<size_t NN, size_t ... Rest>
struct Nums<NN, Rest...> {
  typedef ConsNums<NN, typename Nums<Rest...>::type > type; };

template<> struct Nums<> : public _void {};

#endif
