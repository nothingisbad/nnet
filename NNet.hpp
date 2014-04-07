#ifndef NNET_HPP
#define NNET_HPP
/**
 * @file /home/ryan/programming/nnet/nnet.hpp
 * @author Ryan Domigan <ryan_domigan@sutdents@uml.edu>
 * Created on Apr 06, 2014
 *
 * A simple neural network with compile time constant dimentions.
 */

#include <cmath>
#include <array>
#include <type_traits>
#include <ostream>

template<class Num>
Num sigmoid(const Num& input) {
  static_assert( std::is_floating_point<Num>::value
		 , "sigmoid only makes sense for floating point numbers");
  return 1.0 / (1.0 + exp(-input));
}

/**
 * An N layer neural network where each layer is a std::array of arbitrary (> 1) size.
 * 
 * @tparam Layers: an array of std::array<floating_type, size>
 */
template<class A, class ... Layers>
class NNet {
public:
  typedef typename NNet<Layers ...>::Input Output;
  typedef A Input;
private:
  Input theta;
  Output output;

  NNet<Layers ...> next;
public:
  
  template<class ... Rest>
  NNet(A init, Rest ... rest) : theta(init), next(rest ...) {}

  Output& predict(Input input) {
    for(size_t i = 0, end = input.size();
	i < end; ++i)
      output[i] = sigmoid(input[i] * theta[i]);
    next.predict(output);
  }

  std::ostream& print_theta(std::ostream &out) {
    for(size_t i = 0, end = theta.size();
	i < end; ++i) {
      out << theta.at(i) << "  ";
    }
    out << "\n";
    return next.print_theta(out);
  }
};

template<class A>
class NNet<A> {
  A output;
public:
  typedef A Output;
  typedef A Input;
  
  NNet()  : output() {}

  Output& predict(Input input) {
    output = input;
    return output; 
  }

  std::ostream& print_theta(std::ostream &out) { return out; }
};

#endif
