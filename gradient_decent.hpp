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

#include <boost/iterator.hpp>

#include "./NNet.hpp"

namespace nnet {
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
      static void apply(Net& net, const typename Net::Feed& layer_inputs, AugFeed &activated) {
	using namespace std;
	typedef typename Net::Num Num;
	typedef typename Net::Activation Activation;

	static_assert( is_same<typename Net::tag, NNet_tag>::value
		       , "RMap expects NNet as first argument.");

	static_assert( tuple_size< typename AugFeed::Layer >::value
		       == tuple_size<typename Net::Layer::value_type >::value
		       , "back_propigation: network/feed mismatch" );

	/* map every node with the feed[+1] layer, output to the feed[+0] layer */
	typename Net::NodeInput sum_error{};

	/* For each node, compute the per-weight error of its inputs */
	map_array([&](decltype(net.layer.back()) &node
		      , const Num output_error) {
		    map_array([&](Num &error, Num &nn) {
			error += nn * output_error;
		      }, sum_error, node);
		  }, net.layer
		  , activated.next.layer);

	map_array([&](typename Net::NodeInput& node, const Num delta){
	    map_array([&](Num &activation, Num &nn) {
		nn = activation * delta;
	      }, activated.layer, node);
	  }, net.layer, activated.next.layer);

	activated.layer.back() = sum_error.back() * Activation::bias;
	Map<>::apply([&](//const Num& vv,
			 Num &g_vv, const Num err_next) {
	    g_vv = err_next * Activation::diff(0.0, g_vv); 
	  } //, layer_inputs.layer
	  , activated.layer, sum_error);

      }};  }

  template<class Net, class Augmented>
  void back_propagate(Net& net, typename Net::Feed& input, Augmented &feed) {
    using namespace recurrence_detail;
    RMapLayers<BackPropagate, Net>::apply( net, input, feed);
  }

  /********************************************/
  /*   ____               _ _            _    */
  /*  / ___|_ __ __ _  __| (_) ___ _ __ | |_  */
  /* | |  _| '__/ _` |/ _` | |/ _ \ '_ \| __| */
  /* | |_| | | | (_| | (_| | |  __/ | | | |_  */
  /*  \____|_|  \__,_|\__,_|_|\___|_| |_|\__| */
  /********************************************/
  namespace recurrence_detail {
    struct RegMap {
      template<class Net>
      static void apply(typename Net::Num &reg, Net &net) {
	map_array([&](typename Net::NodeInput &inputs) {
	    Map<0,-1>::apply( [&](typename Net::Num ff) { reg += ff * ff; }, inputs);
	  }, net.layer);
      }};

    struct AverageGrad {
      template<class Net>
      static void apply(Net &grad, Net &theta, float lambda, float m) {
	map_array([&](typename Net::NodeInput& grad_node, typename Net::NodeInput& theta_node) {
	    Map<0,-1>::apply( [&](typename Net::Num &gg, typename Net::Num &tt) {
		gg = (gg + tt * lambda) / m;
	      }, grad_node, theta_node);
	    grad_node.back() = grad_node.back() / m; /* bias */
	  }, grad.layer, theta.layer);
      }};
  }

  template<class Net>
  auto cost_gradient(Net& net
		     , const std::vector< typename Net::Feed::Layer >& training_data
		     , const std::vector< typename Net::Output >& training_lables
		     , float regularization_constant
		     ) -> std::tuple<Net, typename NumType<Net>::type> {
    using namespace recurrence_detail;
    using namespace std;

    typedef typename Net::Feed Feed;
    typedef typename Augment<Feed>::type AugmentedFeed;
    typedef typename NumType<Net>::type Num;
    Net dEdt, dEdt_cumulative{};

    Feed activated, activation_input;
    AugmentedFeed augment;

    size_t m = training_lables.size();
    double J  = 0.0;

    for(size_t i = 0; i < m; ++i) {
      //cout <<"Trainging Y: " << training_lables[i][0] << endl;
      activation_input.layer = training_data[i];
      //predict_with_layer_input(net, activation_input, activated);
      activated.layer = training_data[i];
      predict(net, activated);


      /* I don't need the activations of the output layer, just the error, so I should
	 be OK storing that in the output of the forward feed. */
      map_array([&](Num &h, const Num y) {
	  J += -y * log(h) - (1 - y) * log(1 - h);
	  //J += y > 0.5 ? -log(h) :  -log(1 - h);
	  h = h - y;
	}, activated.output_layer()
	, training_lables[i]);

      augment = Augment<Feed>::apply( activated );

      dEdt = net;
      back_propagate(dEdt, activation_input, augment);

      /* cout<< "augmented activated: "; */
      /* print_feed(augment) << endl; */

      /* accumulate the gradient for each training example */
      map_network([](Num &cume, const Num example) { cume += example;
	}, dEdt_cumulative, dEdt);
    }

    J /= m;

    // print_network(net, cout);

    /* recularize the cost */
    Num reg = 0;
    /* should get  me the sum-squared of the weight coefficients */
    MapLayers< RegMap, Net >::apply(reg, net);
    reg *= regularization_constant / (2 * m);

    J += reg;

    /* take the mean regularized gradient */
    MapLayers<AverageGrad,Net>::apply(dEdt_cumulative, net, regularization_constant, m);

    return make_tuple(dEdt_cumulative, J);
  }

  /****************************/
  /*  _____          _        */
  /* |_   _| __ __ _(_)_ __   */
  /*   | || '__/ _` | | '_ \  */
  /*   | || | | (_| | | | | | */
  /*   |_||_|  \__,_|_|_| |_| */
  /****************************/
  template<class Net>
  void train(Net& net, const std::vector< typename Net::Feed::Layer >& training_data
	     , const std::vector< typename Net::Output >& training_lables
	     , float regularization_constant) {
    auto grad = cost_gradient(net, training_data, training_lables, regularization_constant);

    map_network([](float &nn, float &grad) {
	nn -= grad;
      }, net
      , std::get<0>(grad));
  }

  /************************/
  /*   ____          _    */
  /*  / ___|___  ___| |_  */
  /* | |   / _ \/ __| __| */
  /* | |__| (_) \__ \ |_  */
  /*  \____\___/|___/\__| */
  /************************/
  template<class Net>
  typename NumType<Net>::type
  cost(Net& net
       , const std::vector< typename Net::Feed::Layer >& training_data
       , const std::vector< typename Net::Output >& training_lables
       , typename Net::Num regularization_constant) {
    using namespace recurrence_detail;
    using namespace std;

    typedef typename Net::Feed Feed;
    typedef typename NumType<Net>::type Num;
    Feed forward{};

    size_t m = training_lables.size();
    double J  = 0.0;

    for(size_t i = 0; i < m; ++i) {
      forward.layer = training_data[i];
      predict(net,forward);

      map_array([&](Num &h, const Num y) {
	  J += -y * log(h) - (1 - y) * log(1 - h);
	}, forward.output_layer() , training_lables[i]);
    }
    J /= m;

    /* recularize the cost */
    Num reg = 0;

    /* should get  me the sum-squared of the non-bias weight coefficients */
    MapLayers< RegMap, Net >::apply(reg, net);
    reg *= regularization_constant / (2 * m);

    return   J + reg;
  }

  /****************************/
  /*  ____       _       _    */
  /* |  _ \ _ __(_)_ __ | |_  */
  /* | |_) | '__| | '_ \| __| */
  /* |  __/| |  | | | | | |_  */
  /* |_|   |_|  |_|_| |_|\__| */
  /****************************/
  namespace recurrence_detail {
    /* similar to AugmentedMap */
    struct PrintGradient {
      template<class Net, class Feed>
      static void apply(Net& net, Feed &feed) {
	using namespace std;
	typedef typename NumType<Net>::type Num;

	map_array( [&]( decltype( net.layer.back() )& node
			, Num weighted_error ) {
		     std::cout << "(W " << weighted_error << ")";
		     map_array([&](Num &nn) { std::cout << " " << nn; }, node);
		     std::cout << std::endl;
		   }, net.layer, feed.next.layer);
	cout << "*** Layer ***" << endl;
      }}; }

  /* todo: fix up, this not working well (skipping the bias weights). */
  template<class Net>
  std::ostream& print_gradient(Net& net, typename Net::Feed& grad) {
    using namespace recurrence_detail;
    using namespace std;

    MapLayers<PrintGradient, Net>::apply(net,grad);
    cout << "(W";
    map_array([&](typename Net::Num ff) { cout << " " << ff;}, grad.output_layer());
    cout << ")";
    return std::cout;
  }
}

#endif
