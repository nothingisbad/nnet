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

/***********************************************************/
/*  ___          _   ___              _           _        */
/* | _ ) __ _ __| |_| _ \_ _ ___ _ __(_)__ _ __ _| |_ ___  */
/* | _ \/ _` / _| / /  _/ '_/ _ \ '_ \ / _` / _` |  _/ -_) */
/* |___/\__,_\__|_\_\_| |_| \___/ .__/_\__, \__,_|\__\___| */
/*                              |_|    |___/               */
/***********************************************************/
namespace recurrence_detail {
  struct BackPropagate {
    template<class Net, class AugFeed>
    static void apply(Net& net, AugFeed &feed) {
      using namespace std;
      static_assert( is_same<typename Net::tag, NNet_tag>::value
		     , "RMap expects NNet as first argument.");

      static_assert( tuple_size< typename AugFeed::Layer >::value
		     == tuple_size<typename Net::Layer::value_type >::value
		     , "back_propigation: network/feed mismatch" );

      /* map every node with the feed[+1] layer, output to the feed[+0] layer */
      typename Net::NodeInput accumulate{};

      /* For each node, compute the per-weight error of its inputs*/
      map_array([&](decltype(net.layer.back()) &node
		  , const float output_error) {
		map_array([&](float &acc, float &nn) {
		    acc += nn * output_error;
		  }, accumulate, node);
	      }, net.layer
	      , feed.next.layer);

      map_array([&](typename Net::NodeInput& node, const float delta){
	  map_array([&](float &activation, float &nn) {
	      nn = activation * delta;
	    }, feed.layer, node);
	}, net.layer, feed.next.layer);

      feed.layer.back() = 0.7310585786300049;
      Map<>::apply([&](float &activation, const float err_next) {
	  activation = err_next * (activation * (1 - activation)); 
	}, feed.layer, accumulate);
    }};  }

template<class Net, class Feeds>
void back_propagate(Net& net, Feeds &feed) {
  using namespace recurrence_detail;
  RMapLayers<BackPropagate, Net>::map( net, feed);
}

/******************************************************************/
/*                _        __                  _   _              */
/*   ___ ___  ___| |_     / _|_   _ _ __   ___| |_(_) ___  _ __   */
/*  / __/ _ \/ __| __|   | |_| | | | '_ \ / __| __| |/ _ \| '_ \  */
/* | (_| (_) \__ \ |_    |  _| |_| | | | | (__| |_| | (_) | | | | */
/*  \___\___/|___/\__|___|_|  \__,_|_| |_|\___|\__|_|\___/|_| |_| */
/*                  |_____|                                       */
/******************************************************************/
namespace recurrence_detail {
  struct RegMap {
    template<class Net>
    static void apply(float &reg, Net &net) {
      map_array([&](typename Net::NodeInput &inputs) {
	  Map<0,-1>::apply( [&](float ff) { reg += ff * ff; }, inputs);
	}, net.layer);
    }};

  struct AverageGrad {
    template<class Net>
    static void apply(Net &grad, Net &theta, float lambda, float m) {
      map_array([&](typename Net::NodeInput& grad_node, typename Net::NodeInput& theta_node) {
	  Map<0,-1>::apply( [&](float &gg, float &tt) {
	      gg = (gg + tt * lambda) / m;
	    }, grad_node, theta_node);
	  grad_node.back() = grad_node.back() / m; /* bias */
	}, grad.layer, theta.layer);
    }};
}

template<class Net>
auto cost_function(Net net
		   , const std::vector< typename Net::Feed::Layer >& training_data
		    , const std::vector< typename Net::Output >& training_lables
		    , float regularization_constant
		   ) -> std::tuple<Net, float> {
  using namespace recurrence_detail;
  using namespace std;

  typedef typename Net::Feed Feed;
  typedef typename Augment<Feed>::type AugmentedFeed;
  Net dEdt, dEdt_cumulative;


  Feed forward{}, back{}; 
  AugmentedFeed augment;

  size_t m = training_lables.size();
  double J  = 0.0;

  for(size_t i = 0; i < m; ++i) {
    // cout <<"Trainging Y: " << training_lables[i][0] << endl;
    forward.layer = training_data[i];
    predict(net,forward);
    back = forward;

    map_array([&](const float h, const float y, float &output_err) {
	J -= y * log(h) + (1 - y) * log(1 - h);
	output_err = h - y;
      }, forward.output_layer(), training_lables[i], back.output_layer());

    augment = Augment<Feed>::apply(back);

    dEdt = net;
    back_propagate(dEdt, augment);

    /* accumulate the gradient for each training example */
    map_network([](float &cume, const float example) { cume += example;
      }, dEdt_cumulative, dEdt);
  }

  J /= m;

   // print_network(net, cout);

  /* recularize the cost */
  float reg = 0;
  /* should get  me the sum-squared of the weight coefficients */
  MapLayers< RegMap, Net >::map(reg, net);
  reg *= regularization_constant / (2 * m);

  J += reg;

  /* take the mean regularized gradient */
  MapLayers<AverageGrad,Net>::map(dEdt_cumulative, net, regularization_constant, m);

  return make_tuple(dEdt_cumulative, J);
}

namespace recurrence_detail {
  /* similar to AugmentedMap */
  struct PrintGradient {
    template<class Net, class Feed>
    static void apply(Net& net, Feed &feed) {
      using namespace std;

      map_array( [&]( decltype( net.layer.back() )& node
		    , float weighted_error ) {
		 std::cout << "(W " << weighted_error << ")";
		 map_array([&](float &nn) { std::cout << " " << nn; }, node);
		 std::cout << std::endl;
	       }, net.layer, feed.next.layer);
      cout << "*** Layer ***" << endl;
    }}; }

/* todo: fix up, this not working well (skipping the bias weights). */
template<class Net>
std::ostream& print_gradient(Net& net, typename Net::Feed& grad) {
  using namespace recurrence_detail;
  using namespace std;

  MapLayers<PrintGradient, Net>::map(net,grad);
  cout << "(W";
  map_array([&](float ff) { cout << " " << ff;}, grad.output_layer());
  cout << ")";
  return std::cout;
}

#endif
