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
#include <utility>

template<long Numerator, long Denominator, class Numeric = double>
struct rational_c {

  template<class NewNumeric>
  struct change_value_type { typedef rational_c<Numerator, Denominator, NewNumeric> type; };
  
  typedef Numeric Num;
  constexpr static Num value = (Num)Numerator / (Num)Denominator;
  constexpr static Num value_squared = value * value;
};

template<class Num>
struct Sigmoid {
  typedef Sigmoid<Num> type;

  constexpr static Num apply(const Num& input) {
    static_assert( std::is_floating_point<Num>::value
		   , "sigmoid only makes sense for floating point numbers");
    /* speed up following example at http://msdn.microsoft.com/en-us/magazine/jj658979.aspx */
    //if(input < -45.0) return 0.0;
    //if(input > 45.0) return 1.0;
    return 1.0 / (1.0 + exp(-input));
  }

  /* the input has already had the function applied; caluculate the differntial given that */
  constexpr static Num diff(const Num & g_vv) {
    return g_vv * (1 - g_vv);
  }

  constexpr static Num bias = 0.7310585786300049;
};

template<class Num, class InputMean, class InputVarience>
struct Gaussian {
  typedef typename InputMean::template change_value_type<Num>::type Mean;
  typedef typename InputVarience::template change_value_type<Num>::type Varience;

  typedef Gaussian<Num, Mean, Varience> type;

  Num apply(const Num& vv) {
    static_assert( std::is_floating_point<Num>::value
		   , "Activation functions only makes sense for floating point numbers");
    Num sqr = (vv - Mean::value);
    return exp( - ( (sqr * sqr) / (2 * Varience::value_squared) ) );
  }

  /* given Gaussian applied to the input, compute the deriviative at that point (given g(X)
     compute g'(x) ) */
  Num diff(const Num & g_vv) {
    return - g_vv * sqrt( log(g_vv) ) / (sqrt(2) * Varience::value);
  }
};

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

template<class A>
std::ostream& print_array(const A& aa, std::ostream& out) {
  using namespace std;
  if(aa.size() == 0)
    return cout << "[]";

  cout << "[" << aa[0];
  Map<1>::apply([&](float ff) { out << ", " << ff; }, aa);
  return cout << "]";
}

template<class T, size_t N>
std::ostream& operator<<(const std::array<T, N>& aa, std::ostream& out) {
  return print_array(aa, out);
}

template<class A>
std::ostream& print_array(const A& aa) { return print_array(aa, std::cout); }

/*****************************/
/*  _____            _       */
/* |_   _|   _ _ __ | | ___  */
/*   | || | | | '_ \| |/ _ \ */
/*   | || |_| | |_) | |  __/ */
/*   |_| \__,_| .__/|_|\___| */
/*            |_|            */
/*****************************/
template<class FN, class Tup, size_t elem = 0, size_t last = std::tuple_size<Tup>::value>
struct MapTuple {
  static void apply(Tup& tup) {
    FN::apply( std::get<elem>(tup) );
    MapTuple<FN,Tup,elem + 1, last>::apply(tup);
  } };

template<class FN, class Tup, size_t elem>
struct MapTuple<FN, Tup, elem, elem> { static void apply(Tup&) {} };


template<class FN, class Tup>
void map_tuple(FN&&, Tup& tup) { MapTuple<FN, typename std::remove_reference<Tup>::type >::apply(tup); }


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

namespace detail {
  template<class T> struct GetDepth : public intC<T::depth + 1>::type {};

  template<> struct GetDepth<_void> : public intC<0>::type {};
}

/**********************************/
/*  __  __ ____  _     _     _    */
/* |  \/  |  _ \| |   (_)___| |_  */
/* | |\/| | |_) | |   | / __| __| */
/* | |  | |  __/| |___| \__ \ |_  */
/* |_|  |_|_|   |_____|_|___/\__| */
/**********************************/
/* meta-programming-list; a list of types which can be instantiated to make a sort of tuple.
   Having said that; maybe I should generate a std::tuple and use the boost libs.  Well, I've already got this going;
   so sticking with it for now
   */
struct MPList_tag { typedef MPList_tag tag_type; };

template<class List> struct Car : public Identity<typename List::Car> {};
template<> struct Car<_void> : public _void {};

template<class List> struct Cdr : public Identity<typename List::Cdr> {};
template<> struct Cdr<_void> : public _void {};

template<class T, class Next>
struct Cons : public MPList_tag {
  typedef Cons<T, Next> type;

  typedef T Car;
  typedef Next Cdr;

  /* I can use the instatiated list at run-time */
  T value;
  Next next;
  
  static const size_t depth = detail::GetDepth<Next>::value;
};

template<class ... List> struct MPList;

template<class Head, class ... Tail>
struct MPList<Head, Tail...> : public Cons<Head, typename MPList<Tail...>::type> {};

template<> struct MPList<> : public _void {};

template<class Fn, class Init, class InputList>
struct Fold {
  static_assert( std::is_same<typename InputList::type::tag_type, MPList_tag>::value
		 , "Length requires an MPList or Cons cell.");
  typedef typename InputList::type List;
  typedef typename
  Fn::template Apply<typename Car<List>::type
		     , typename Fold<Fn, Init, typename Cdr<List>::type>::type
		     > type;
};

template<class Fn, class Init>
struct Fold<Fn,Init,_void> { typedef typename Init::type type; };

template<class Flatten, class List>
struct map_MPList {
  template<class Fn>
  void apply(Fn fn, List& ll) {
    Flatten::apply(fn, ll);

    map_MPList<Flatten, typename Cdr<List>::type
	       >::apply(fn, ll.next);
  }
};


template<class List>
struct Length {
  static_assert( std::is_same<typename List::tag_type, MPList_tag>::value
		 , "Length requires an MPList or Cons cell.");
  const static size_t value = List::depth;
};

template<size_t N, class NNs>
struct ConsNums {
  typedef NNs Next;
  static const size_t value = N;
  static const size_t depth = detail::GetDepth<Next>::value;

  typedef ConsNums<N, NNs> type;
};

template<size_t ... Rest> struct Nums;

template<size_t NN, size_t ... Rest>
struct Nums<NN, Rest...> {
  typedef ConsNums<NN, typename Nums<Rest...>::type > type; };

template<> struct Nums<> : public _void {};

template<class List>
struct LengthMPList : public intC<List::depth> {};

/*********************************/
/*  ____                         */
/* |  _ \ __ _ _ __   __ _  ___  */
/* | |_) / _` | '_ \ / _` |/ _ \ */
/* |  _ < (_| | | | | (_| |  __/ */
/* |_| \_\__,_|_| |_|\__, |\___| */
/*                   |___/       */
/*********************************/
struct IncItr { template<class T> static void apply(T&& t) { ++t; } };

template<class ... Types>
struct RangeItr {
  typedef std::tuple<Types ...> value_type;
  value_type _itrs;

  RangeItr( const Types& ... input ) : _itrs(input ...) {}

  RangeItr& operator++() {
    map_tuple(IncItr(), _itrs);
    return *this;
  }
  value_type& operator*() { return _itrs; }

  /* note: only compairs the _first_ element (so I can have ranges of different sizes; stick the shortest range in
     the first position) */
  bool operator!=(const RangeItr<Types...>& itr) {
    return std::get<0>(itr._itrs) != std::get<0>(_itrs);
  }
};

template<class ... Pairs>
struct Range {
  typedef RangeItr<typename Pairs::first_type ...> iterator_type;
  RangeItr<typename Pairs::first_type ...> _begin, _end;

  iterator_type& begin() { return _begin; }
  iterator_type& end() { return _end; }
  const iterator_type& begin() const { return _begin; }
  const iterator_type& end() const { return _end; }

  Range(const Pairs& ...  pp) : _begin(pp.first ...), _end(pp.second ...) {}

  Range() = delete;
  Range(const Range&) = default;
  ~Range() = default;
};

template<class Pairs>
struct Range<Pairs> {
  typedef typename Pairs::first_type iterator;
  iterator _begin, _end;

  iterator& begin() { return _begin; }
  iterator& end() { return _end; }
  const iterator& begin() const { return _begin; }
  const iterator& end() const { return _end; }

  Range(const Pairs& pp) : _begin(pp.first), _end(pp.second) {}

  Range() = delete;
  Range(const Range&) = default;
  ~Range() = default;
};

template<class ... PType>
Range< PType ... > make_range(const PType& ... pairs) {
  return Range< PType ... >( pairs ...);
}

template<class Collection>
std::pair<typename Collection::iterator, typename Collection::iterator>
coll_pair(Collection& cc) {  return std::make_pair(cc.begin(), cc.end()); }

#endif
