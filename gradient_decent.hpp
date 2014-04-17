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

      cout << "Backpropigating on network: \n";
      print_network(net) << endl;

      /* map every node with the feed[+1] layer, output to the feed[+0] layer */
      typename Net::NodeInput accumulate{};

      cout << "Feed: \n";
      print_feed(feed) << endl;

      /* For each node, compute the per-weight error of its inputs*/
      ref_map([&](decltype(net.layer.back()) &node
		  , const float output_error) {
		ref_map([&](float &acc, float &nn) {
		    acc += nn * output_error;
		  }, accumulate, node);
	      }, net.layer
	      , feed.next.layer);

      cout << "should be my d_2 prefix:\n";
      print_array(accumulate) << endl;

      /* gets store d(n) for use by the enclosing recurance */
      cout << "should be my d_2 suffix input:\n";
      print_array(feed.layer) << endl;


      ref_map([&](typename Net::NodeInput& node, const float delta){
	  ref_map([&](float &activation, float &nn) {
	      nn = activation * delta;
	    }, feed.layer, node);
	}, net.layer, feed.next.layer);

      feed.layer.back() = 0.7310585786300049;
      Map<>::apply([&](float &activation) {
	  activation = (activation * (1 - activation)); 
	}, feed.layer);

      cout << "should be my d_2 suffix:\n";
      print_array(feed.layer) << endl;

      Map<>::apply([&](float &activation, const float err_next) {
	  activation = err_next * activation; 
	}, feed.layer, accumulate);

      cout << "should be my d_2:\n";
      print_array(feed.layer) << endl;

      cout << "Theta: \n";
      ref_map([&](typename Net::NodeInput& node) {
	  ref_map([&](float nn) {
	      cout << " " << nn;
	    }, node);
	  cout << "\n";
	}, net.layer);
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
      ref_map([&](typename Net::NodeInput &inputs) {
	  Map<0,-1>::apply( [&](float ff) {
	      // std::cout << "  ff: " << ff << std::endl;
	      reg += ff * ff;
	    }, inputs);
	}, net.layer);
    }}; }

template<class Net, class ErrorFn>
auto cost_function(Net& net
		   , const std::vector< typename Net::Feed::Layer >& training_data
		    , const std::vector< typename Net::Output >& training_lables
		    , ErrorFn error
		    , float regularization_constant
		   ) -> std::tuple<Net, float>{
  using namespace recurrence_detail;
  using namespace std;

  static_assert( is_same<bool, decltype(error(1.0f,1.0f))>::value
		 , "error function is expected to return bool (true if considered a label match)");

  typedef typename Net::Feed Feed;
  typedef typename Augment<Feed>::type AugmentedFeed;
  Net dEdt, dEdt_cumulative;


  Feed forward{}, back{}; 
  AugmentedFeed augment;

  size_t m = training_lables.size();
  double J  = 0.0;

  for(size_t i = 0; i < m; ++i) {
    dEdt = net;

    // cout <<"Trainging Y: " << training_lables[i][0] << endl;
    forward.layer = training_data[i];
    predict(net,forward);
    back = forward;


    ref_map([&](const float h, const float y, float &output_err) {
	J -= error(h, y) ? log(h) : log(1 - h);
	// cout << "h: " << h << " y: " << y << " incremental cost : " <<  (error(h, y) ? log(h) : log(1 - h)) << endl;
	output_err = h - y;
      }, forward.output_layer(), training_lables[i], back.output_layer());

    augment = Augment<Feed>::apply(back);
    /* cout << "Augmented feed-forward:"; */
    /* print_feed(augment) << "\n"; */

    /* cout << "Based on:"; */
    /* print_feed(back) << "\n"; */

    back_propagate(dEdt, augment);

    /* cout << "Back:"; */
    /* print_feed(back) << "\n"; */

    /* cout << "Back Grad:\n"; */
    /* print_network(dEdt) << "\n"; */

    /* accumulate the gradient for each training example */
    map([](float &aa, float &bb) {
	aa += bb;
      }, dEdt_cumulative, dEdt);
  }

  J /= m;

  map([&m](float &ff) { ff /= m; }, dEdt);
  // print_network(net, cout);

  /* recularize the cost */
  float reg = 0;
  /* should get  me the sum-squared of the weight coefficients */
  MapLayers< RegMap, Net >::map(reg, net);

  J += reg;

  /* take the mean regularized gradient */
  map([&](float &grad, float &theta) {
      grad = (grad + theta * regularization_constant) / m;
    }, dEdt_cumulative, net);

  return make_tuple(dEdt_cumulative, J);
}

namespace recurrence_detail {
  /* similar to AugmentedMap */
  struct PrintGradient {
    template<class Net, class Feed>
    static void apply(Net& net, Feed &feed) {
      using namespace std;

      ref_map( [&]( decltype( net.layer.back() )& node
		    , float weighted_error ) {
		 std::cout << "(W " << weighted_error << ")";
		 ref_map([&](float &nn) { std::cout << " " << nn; }, node);
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
  ref_map([&](float ff) { cout << " " << ff;}, grad.output_layer());
  cout << ")";
  return std::cout;
}

#endif
