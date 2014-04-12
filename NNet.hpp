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
#include <iostream>
#include <random>
#include <type_traits>

#include "./aggregation.hpp"

template<class Num>
Num sigmoid(const Num& input) {
  static_assert( std::is_floating_point<Num>::value
		 , "sigmoid only makes sense for floating point numbers");
  /* speed up following example at http://msdn.microsoft.com/en-us/magazine/jj658979.aspx */
  /* if(input < -45.0) return -1.0; */
  /* if(input > 45.0) return 1.0; */
  return 1.0 / (1.0 + exp(-input));
}

struct _void { typedef _void type; };

/****************************************/
/*  _____             _ _   _      _    */
/* |  ___|__  ___  __| | \ | | ___| |_  */
/* | |_ / _ \/ _ \/ _` |  \| |/ _ \ __| */
/* |  _|  __/  __/ (_| | |\  |  __/ |_  */
/* |_|  \___|\___|\__,_|_| \_|\___|\__| */
/****************************************/
struct FeedNet_tag;
struct TagBase {};

template<class A, class ...Layers>
struct FeedNet : public TagBase {
  typedef FeedNet_tag tag;

  typedef FeedNet<Layers ...> Next;
  typedef A Layer;
  typedef typename Next::Output Output;

  static constexpr bool Recursive = true;

  Layer layer;
  Next next;
  
  Output& output_layer() { return next.output_layer(); }
};
  
/* Base class */
template<class A>
struct FeedNet<A> : public TagBase {
  typedef FeedNet_tag tag;

  typedef _void Next;
  typedef A Layer;
  typedef A Output;

  static constexpr bool Recursive = false;

  Layer layer;

  Output& output_layer() { return layer; }
};


/**************************/
/*  _   _ _   _      _    */
/* | \ | | \ | | ___| |_  */
/* |  \| |  \| |/ _ \ __| */
/* | |\  | |\  |  __/ |_  */
/* |_| \_|_| \_|\___|\__| */
/**************************/
/**
 * An N layer neural network where each layer is a std::array of arbitrary (> 1) size.
 * 
 * @tparam Layers: an array of std::array<floating_type, size>
 */
struct NNet_tag;

template<class A, class ... Layers>
class NNet : public TagBase {
public:
  typedef NNet_tag tag;
  
  typedef FeedNet<A, Layers ...> FeedType;
  typedef NNet<Layers ...> Next;

  typedef typename NNet<Layers ...>::Input LayerOutput;
  typedef typename NNet<Layers ...>::Output Output;
  typedef A Input;

  constexpr static size_t depth = Next::depth + 1;

  /* AugmentedInput includes a bias element of +1 with each node weights */
  typedef std::array<float, std::tuple_size<A>::value + 1 > NodeInput;
  typedef std::array< NodeInput, std::tuple_size< LayerOutput >::value > Theta;

  Theta theta;
  Next next;
  
  static constexpr size_t node_size() { return std::tuple_size< LayerOutput >::value; }

  /* counts the bias */
  static constexpr size_t node_input_size(size_t layer) { return std::tuple_size< NodeInput >::value; }

  template<class ... Rest>
  NNet(A init, Rest ... rest) : theta(init), next(rest ...) {}
  NNet() : theta{} {}
};

/* base case.  I one layer NNet doesn't make sense */
template<class A, class B>
class NNet<A, B> {
public:
  
  typedef FeedNet<A, B> FeedType;
  typedef _void Next;

  typedef B LayerOutput;
  typedef B Output;
  typedef A Input;

  constexpr static size_t depth = 1;

  /* AugmentedInput includes a bias element of +1 with each node weights */
  typedef std::array<float, std::tuple_size<A>::value + 1 > NodeInput;
  typedef std::array< NodeInput, std::tuple_size< LayerOutput >::value > Theta;

  Theta theta;
  Next next;
  
  static constexpr size_t node_size() { return std::tuple_size< LayerOutput >::value; }
  static constexpr size_t node_input_size(size_t layer) { return std::tuple_size< NodeInput >::value; }

  template<class ... Rest>
  NNet(A init, Rest ... rest) : theta(init) {}
  NNet() : theta{} {}
};


template<class T>
struct TypeGetNext {
  typedef T type;
  static T&& apply(T &&input) { return std::forward<T>(input); }
};

template<class T>
struct ClassGetNext {
  typedef typename T::Next type;
  static typename T::Next&& apply(T &&input) { return std::move(input.next); }
};

template<class T>
struct GetNext
  : public std::conditional< std::is_base_of<T, TagBase>::value
			     , ClassGetNext<T>
			     , TypeGetNext<T>
			     >::type {};

template<class T>
typename GetNext<T>::type&& next(T &&input) {
  return std::forward< typename GetNext<T>::type >( GetNext<T>::apply(input) );
}


/**************************************/
/*  _____     _     _ _   _      _    */
/* |  ___|__ | | __| | \ | | ___| |_  */
/* | |_ / _ \| |/ _` |  \| |/ _ \ __| */
/* |  _| (_) | | (_| | |\  |  __/ |_  */
/* |_|  \___/|_|\__,_|_| \_|\___|\__| */
/**************************************/
template<class LayerFn, class NetType>
class FoldLayers {
public:
  template<class ... Aux>
  static void map(Aux&& ... aux) {
    LayerFn::apply(aux...);
    FoldLayers<LayerFn, typename NetType::Next>::map( next(aux) ...);
  }
};

template<class LayerFn>
class FoldLayers<LayerFn,_void> {
public:
  template<class ... Aux> static void map(Aux...) { }
};


/**
 * Fold a net with static functions
 * 
 * @tparam Net:
 */
template<class NetType>
class FoldNet {
public:
  typedef NetType Net;
  typedef typename Net::FeedType Feed;
  typedef FoldNet<typename Net::Next> Next;

private:
  template<class Fn, class Output>
  static void rmap_helper(Fn fn
			  , Net& net
			  , const typename Feed::Next &feed
			  , Output& output) {
    size_t idx = 1;

    
    /* Apply< typename NextArgs<void (*)(Fn,Net,const typename Feed::Next&, decltype(feed.layer)) >::type */
    /* 	   , &Next::rmap_helper >::void_do_it; */
    Next::rmap_helper(fn, feed.next, net.next, feed.layer);

    net.fold_nodes([&](float theta) {
	output[idx] = idx ? fn(theta, 1) : fn(theta, feed.layer[idx]);
	++idx;
      },
      [&]() { idx = 1; } );
  }

public:
  /**
   * Augments each row of the input array with an initial '1' and folds
   * with a network of matching dimentions.
   *
   * Results are stored in the FeedNet
   */
  template<class Fn>
  static void augment_map(Fn fn, Feed& feed, const Net& net) {
    typename Feed::Next::Layer& output = feed.next.layer;

    ::fold( [&](const typename Net::NodeInput& node) {
	size_t idx = 0;
	::fold([&](float theta) {
	    output[idx] = idx ? fn(theta, 1) : fn(theta, feed.layer[idx]);
	    ++idx;
	  }, node);
      }, net.theta);

    Next::argument_map(fn, feed.next, net.next);
  }

  /**
   * Maps from deep layers onto shallow.
   * 
   * Note drops node[0] (the bias weight) from the neural net to make
   * the dimentions match
   * 
   * @tparam Fn:
   * @tparam Feeds:
   * @return: 
   */
  template<class Fn, class Feeds>
  static void rmap(Fn fn, Net& net, const Feeds& feed) {
    rmap_helper(fn, net, feed.next, feed.layer);
  }

  template<class Fn>
  static void map(const Fn fn, Net &aa, const Net &bb) {
    for(size_t node = 0; node < Net::node_size(); ++node) {
      for(size_t weight = 0; weight < Net::node_input_size(); ++weight) {
	aa.theta[node][weight] = fn( aa.theta[node][weight], bb.theta[node][weight]);
      }}
    Next::map(fn, aa, bb);
  }


  template<class Fn>
  static void map(Fn fn, Net& net) {
    for(size_t i = 0, end = std::tuple_size< typename Net::LayerOutput >::value;
	i < end; ++i) {
      for(size_t j = 0, end_j = std::tuple_size< typename Net::Input >::value;
	  j < end_j; ++j)
	net.theta[i][j] = fn( net.theta[i][j] );
    }
    Next::map(fn, net.next);
  }


  template<class Fn>
  static void fold(Fn fn, const Net& net) {
    // Next::fold(fn, net.next);
  }

  template<class Fn, class FnLayer>
  static void fold(Fn fn, FnLayer fn_layer, const Net& net) {
    fold(fn, net);
    fn_layer();

    Next::fold(fn,fn_layer, net.next);
  }

  static typename Net::LayerOutput&& layer_predict(const Net& net, const typename Net::Input& input) {
    typename Net::LayerOutput output{};
    for(size_t i = 0, end = std::tuple_size< typename Net::LayerOutput >::value;
	i < end; ++i) {
      output[i] = net.theta[i][0];
      for(size_t j = 1, end = std::tuple_size< typename Net::Input >::value;
	  j < end; ++j)
	output[i] += input[j] * net.theta[i][j];

      output[i] = sigmoid(output[i]);
    }

    return std::move(output);
  }

  static typename Net::Output&& predict(const Net& net, const typename Net::Input& input) {
    return std::move( Next::predict(net.next, layer_predict(net, input)) );
  }
};

/* base class */
template<>
class FoldNet< _void > {
public:
  template<class Fn, class FnLayer> static void fold(Fn, FnLayer, _void) {}
  template<class Fn> static void fold(Fn,  _void) {}

  template<class Net, class Input>
  static Input&& predict(const Net& net, const Input& input) {
    return std::move( Input(input) );
  }
};

/*************************************************/
/*  _____                 _   _                  */
/* |  ___|   _ _ __   ___| |_(_) ___  _ __  ___  */
/* | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __| */
/* |  _|| |_| | | | | (__| |_| | (_) | | | \__ \ */
/* |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/ */
/*************************************************/
/*********************************************/
/*  ___                                      */
/* | _ \___ __ _  _ _ _ _ _ ___ _ _  __ ___  */
/* |   / -_) _| || | '_| '_/ -_) ' \/ _/ -_) */
/* |_|_\___\__|\_,_|_| |_| \___|_||_\__\___| */
/*********************************************/
namespace recurrence_detail {
  struct FoldEveryWeight {
    template<class Fn, class LayerFn, class Net>
    static void apply(Fn fn, LayerFn layer_fn, Net& net) {
      ::fold( [&](decltype(net.theta[0])&& weights) {
	  ::fold(fn, weights);
	  layer_fn();
	}, net.theta);
    }

    template<class Fn, class Theta>
    static void apply(Fn fn, Theta& theta) {
      ::fold( [&](const decltype(theta[0])& weights) {
	  ::fold(fn, weights);
	}, theta);
    }};
}

template<class Fn, class Net>
void fold(Fn fn, const Net& net) {
  using namespace recurrence_detail;
  FoldLayers<FoldEveryWeight, Net>::fold(fn, net);
}

template<class Fn, class LayerFn, class Net>
void fold(Fn fn, LayerFn layer_fn, const Net& net) {
  using namespace recurrence_detail;
  FoldLayers<FoldEveryWeight, Net>::map(fn, layer_fn, net);
}


template<class Net>
std::ostream& print_theta(const Net& net, std::ostream &out) {
  using namespace std;

  fold([&](float f) { out << " " << f; }
      , [&]() { out << endl; }
      , net);

  return out;
}

template<class Net>
std::istream& read_theta(Net& net, std::istream &in) {
  using namespace std;
  net.map([&](float& f) { in >> f; });
  return in;
}

template<class Net>
void permute(Net &n, float low, float high) {
  using namespace std;
  random_device rd;
  mt19937 gen(rd());

  uniform_real_distribution<float> dist(low, high);

  FoldNet<Net>::map([&](float f) {
      return f += dist(gen);
    }, n);
}

#endif
