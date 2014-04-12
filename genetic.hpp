#ifndef GENETIC_HPP
#define GENETIC_HPP
/**
 * @file /home/ryan/programming/nnet/genetic.hpp
 * @author Ryan Domigan <ryan_domigan@sutdents@uml.edu>
 * Created on Apr 07, 2014
 *
 * An algorithm for evolving a neural net.  I would like to sample weights from
 * several successful algorithms, but for now that's not what I'm doing.
 */

#include <random>
#include <algorithm>
#include <iostream>

#include "./NNet.hpp"

template<class NetType, size_t num_networks>
class Evolve {
public:
  typedef std::array<float, num_networks> WeightType;
private:
  /* from examples at cppreference.com */
  std::random_device _rd;
  std::mt19937 _gen;

  std::array<NetType, num_networks> _networks;
  WeightType _weights
    , _weight_sums;

  constexpr static float _bump_size = 1.0 / (float)num_networks;

  void re_compute_weights() {
    _weight_sums[0] = _weights[0];

    for(size_t i = 1; i < num_networks; ++i)
      _weight_sums[i] = _weight_sums[i - 1] + _weights[i];

    float normal = 1 / _weight_sums[num_networks - 1];
    
    for(size_t i = 0; i < num_networks; ++i)
      _weight_sums[i] *= normal;
  }
  
  /* index of the lowest weight */
  size_t min_weight_index() {
    size_t min = 0;
    float min_val = 1;
    for(size_t i = 0; i < num_networks; ++i) {
      if(_weights[i] < min_val) {
	min = i;
	min_val = _weights[i];
      }
    }

    return min;
  }

  float average_weight() {
    float result = 0.0;
    for(size_t i = 0; i < num_networks; ++i)
      result += _weights[i];
    return result / num_networks;
  }

  /* index of networks which are currently being evaluated */
  int _porgeny, _current_heigh;
public:
  void rank(int n, float weight) {
    _weights[n] += weight;
    re_compute_weights();
  }
  
  void set_weights(const WeightType&& ws) {
    _weights = ws;
    re_compute_weights();
  }

  void bump_up(size_t i) {
    _weights[i] += _bump_size;
    re_compute_weights();
  }

  void bump_down(size_t i) {
    _weights[i] -= _bump_size;
    re_compute_weights();
  }

  /* mutate the lowest weighted neural net */
  void mutate() {
    size_t idx = min_weight_index();
    _weights[idx] = average_weight();
    at(idx).permute(-2, 2);

    re_compute_weights();
  }

  const NetType& at(size_t i) const { return _networks[i]; }

  NetType& at(size_t i) { return _networks[i]; }

  /* picks an index by taking a weighted random sample */
  size_t sample() {
    /* binary search */
    float key =  std::generate_canonical<float,10>(_gen);
    //cout << "Value: "
    size_t high = num_networks
      , low = 0
      , mid;

    while(high >= low) {
      mid = (high + low) / 2;

      if( _weight_sums[mid] >= key
	  && (mid == 0 		/* don't test mid - 1 (unsigned int will roll-over) */
	      || _weight_sums[mid - 1] < key))
	return mid;

      if( _weight_sums[mid] < key )
	low = mid + 1;

      else high = mid - 1;
    }
    return -1; 			/* should not be reachable */
  }

  /****************************************************************/
  /*   ____                _                   _                  */
  /*  / ___|___  _ __  ___| |_ _ __ _   _  ___| |_ ___  _ __ ___  */
  /* | |   / _ \| '_ \/ __| __| '__| | | |/ __| __/ _ \| '__/ __| */
  /* | |__| (_) | | | \__ \ |_| |  | |_| | (__| || (_) | |  \__ \ */
  /*  \____\___/|_| |_|___/\__|_|   \__,_|\___|\__\___/|_|  |___/ */
  /****************************************************************/
  Evolve() :  _gen( _rd() ) {
    for(size_t i = 0; i < _weights.size(); ++i) {
      _weights[i] = _bump_size;
      _networks[i].random_init(-1,1);
    }

    re_compute_weights();
  }

  /******************************************/
  /*  ____       _       _   _              */
  /* |  _ \ _ __(_)_ __ | |_(_)_ __   __ _  */
  /* | |_) | '__| | '_ \| __| | '_ \ / _` | */
  /* |  __/| |  | | | | | |_| | | | | (_| | */
  /* |_|   |_|  |_|_| |_|\__|_|_| |_|\__, | */
  /*                                 |___/  */
  /******************************************/
  /* prints the weighted values for the sampling
     of indecies */
  std::ostream& print_weights(std::ostream &out) {
    out << "[" << _weight_sums[0];
    std::for_each(_weight_sums.begin() + 1, _weight_sums.end()
		  , [&out](float ff) { out << ", " << ff; });
    return out << "]";
  }
};


#endif
