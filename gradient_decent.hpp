#ifndef GRADIENT_DECENT_HPP
#define GRADIENT_DECENT_HPP
/**
 * @file /home/ryan/programming/nnet/gradient_decent.hpp
 * @author Ryan Domigan <ryan_domigan@sutdents@uml.edu>
 * Created on Apr 08, 2014
 *
 * Perform gradient decent to learn 
 */
#include <vector>
#include "./NNet.hpp"

/********************************************/
/*   ____               _ _            _    */
/*  / ___|_ __ __ _  __| (_) ___ _ __ | |_  */
/* | |  _| '__/ _` |/ _` | |/ _ \ '_ \| __| */
/* | |_| | | | (_| | (_| | |  __/ | | | |_  */
/*  \____|_|  \__,_|\__,_|_|\___|_| |_|\__| */
/********************************************/
template<class NetType, class ErrorFn>
const NetType&& gradient(const NetType& net
			 , const typename NetType::Input& input
			 , const typename NetType::Output& labels
			 , const ErrorFn error) {
  using namespace std;
  typedef typename NetType::FeedType FeedType;
  typedef FoldNet<NetType> FoldNet;

  NetType dEda = net;
  FeedType a;

  /* feed forward */
  FoldNet::augment_map([](float theta, float input) {
      return sigmoid(theta * input); }
    , a, net);

  /* back propigate */
  ref_map([&](float predicted, float label) {
      return error(predicted, label);
    }
    , a.output_layer()
    , labels);

  FoldNet::rmap( [](float weight, float err) {
      return weight * err;
    }
    , dEda, a);

  return move(dEda);
}

/******************************************************************/
/*                _        __                  _   _              */
/*   ___ ___  ___| |_     / _|_   _ _ __   ___| |_(_) ___  _ __   */
/*  / __/ _ \/ __| __|   | |_| | | | '_ \ / __| __| |/ _ \| '_ \  */
/* | (_| (_) \__ \ |_    |  _| |_| | | | | (__| |_| | (_) | | | | */
/*  \___\___/|___/\__|___|_|  \__,_|_| |_|\___|\__|_|\___/|_| |_| */
/*                  |_____|                                       */
/******************************************************************/
template<class NetType, class ErrorFn>
float cost_function(const NetType& net
		    , const typename NetType::Input& input
		    , const typename NetType::Output& training_lables
		    , float regularization_constant
		    , ErrorFn error
		    ) {
  typedef FoldNet<NetType> Fold;
  size_t m = input.size();

  /************************/
  /*   ____          _    */
  /*  / ___|___  ___| |_  */
  /* | |   / _ \/ __| __| */
  /* | |__| (_) \__ \ |_  */
  /*  \____\___/|___/\__| */
  /************************/
  double J  = 0.0, h;
  typename NetType::Output prediction = Fold::predict(net, input);

  for(size_t k = 0; k < std::tuple_size< typename NetType::Output >::value; ++k) {
      h = log( prediction[k] );
      J -= error(prediction[k], training_lables[k]) ? log(h) : (1 - log(h));
  }

  J /= (double)m;


  double reg = 0;
  Fold::fold([&](float f) -> void {
      reg += f * f;
    }, net);
  reg *= (regularization_constant / (2 * m));

  return J + reg;
}


#endif
