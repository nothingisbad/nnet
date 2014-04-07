/**
 * @file /home/ryan/programming/nnet/test_NNet.cpp
 * @author Ryan Domigan <ryan_domigan@sutdents@uml.edu>
 * Created on Apr 06, 2014
 */

#include "NNet.hpp"
#include "genetic.hpp"

#include <iostream>

template<size_t i, class Num, class ... Args>
struct MakeArray {
  typedef typename MakeArray<i+1,Args...>::return_type return_type;
  static return_type&& a(Num n, Args ... as) {
    return_type &&tmp = MakeArray<i + 1, Args ...>::a(as ... );
    tmp[i - 1] = static_cast<float>(n);
    return std::move(tmp);
  }
};

template<size_t i, class Num>
struct MakeArray<i, Num> {
  typedef std::array<float,i> return_type;

  static std::array<float,i> a(Num n) {
    std::array<float,i> tmp;
    tmp[i - 1] = static_cast<float>(n);
    return tmp;
  }
};

template<class ... args>
typename MakeArray<1, args ...>::return_type make_array(args ... a) {
  return MakeArray<1, args ...>::a(a ...);
}

int main() {
  using namespace std;
  /* NNet< array<float,3> */
  /* 	, array<float,2> */
  /* 	, array<float,2> > */
  /*   net( make_array(1,2,3) */
  /*        , make_array(1,2) ); */
  
  /* net.print_theta(cout); */

  Evolve< NNet< array<float,3> >
	  , 3 > e;
  e.set_weights( make_array(1,2,3) );

  e.print_weights(cout) << endl;

  int count[3] = {0,0,0};
  
  for(int i = 0; i < 1000; ++i)
    count[ e.sample() ] += 1;

  for(int i = 0; i < 3; ++i)
    cout << "Sampled " << i << " " << count[i] << " times " << endl;

  return 0;
}
