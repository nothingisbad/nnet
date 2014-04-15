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
#include <utility>

#include "./aggregation.hpp"

template<class Num>
Num sigmoid(const Num& input) {
  static_assert( std::is_floating_point<Num>::value
		 , "sigmoid only makes sense for floating point numbers");
  /* speed up following example at http://msdn.microsoft.com/en-us/magazine/jj658979.aspx */
  if(input < -45.0) return 0.0;
  if(input > 45.0) return 1.0;
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

  Layer layer;
  Next next;
  
  Output& output_layer() { return next.output_layer(); }

  FeedNet() : layer{} {}
  FeedNet(float n) : next(n) { ref_map([&](float &f) { f = n; }, layer); }
  FeedNet(const FeedNet &feed) : layer(feed.layer), next(feed.next) {}
};
  
/* Base class */
template<class A>
struct FeedNet<A> : public TagBase {
  typedef FeedNet_tag tag;

  typedef _void Next;
  typedef A Layer;
  typedef A Output;

  _void next;
  Layer layer;

  Output& output_layer() { return layer; }

  FeedNet() : layer{} {}
  FeedNet(float n) { ref_map([&](float &f) { f = n; }, layer); }
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
  
  typedef FeedNet<A, Layers ...> Feed;
  typedef NNet<Layers ...> Next;

  typedef typename NNet<Layers ...>::Input LayerOutput;
  typedef typename NNet<Layers ...>::Output Output;
  typedef A Input;

  constexpr static size_t depth = Next::depth + 1;

  /* AugmentedInput includes a bias element of +1 with each node weights */
  typedef std::array<float, std::tuple_size<A>::value + 1 > NodeInput;
  typedef std::array< NodeInput, std::tuple_size< LayerOutput >::value > Layer;

  Next next;
  Layer layer;
  
  static constexpr size_t node_size() { return std::tuple_size< LayerOutput >::value; }

  /* counts the bias */
  static constexpr size_t node_input_size(size_t layer) { return std::tuple_size< NodeInput >::value; }

  template<class ... Rest>
  NNet(const Layer& init, Rest ... rest) :  next(rest ...) , layer(init) {}
  NNet(const NNet& net) : next(net.next), layer(net.layer) {}
  NNet() : layer{} {}
};

/* base case.  I one layer NNet doesn't make sense */
template<class A, class B>
class NNet<A, B> {
public:
  typedef NNet_tag tag;
  
  typedef FeedNet<A, B> Feed;
  typedef _void Next;

  typedef B LayerOutput;
  typedef B Output;
  typedef A Input;

  constexpr static size_t depth = 1;

  /* AugmentedInput includes a bias element of +1 with each node weights */
  typedef std::array<float, std::tuple_size<A>::value + 1 > NodeInput;
  typedef std::array< NodeInput, std::tuple_size< LayerOutput >::value > Layer;

  Next next;
  Layer layer;
  
  static constexpr size_t node_size() { return std::tuple_size< LayerOutput >::value; }
  static constexpr size_t node_input_size(size_t layer) { return std::tuple_size< NodeInput >::value; }

  NNet(const Layer& init) : layer(init) {}
  NNet(const NNet& net) : next(net.next), layer(net.layer) {}
  NNet() : layer{} {}
};

/*************************************************/
/*  _____                 _   _                  */
/* |  ___|   _ _ __   ___| |_(_) ___  _ __  ___  */
/* | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __| */
/* |  _|| |_| | | | | (__| |_| | (_) | | | \__ \ */
/* |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/ */
/*************************************************/
namespace recurrence_detail {
  template<class T>
  struct TypeGetNext {
    typedef T Next;
    static T&& apply(T &&input) {
      return std::forward<T>(input);
    }};

  template<class T>
  struct ClassGetNext {
    typedef typename T::Next Next;
    static Next&& apply(T&& input) {
      return std::forward<Next>( input.next );
    }};

  template<class T>
  struct GetNext
    : public std::conditional< std::is_class<T>::value && std::is_base_of<TagBase ,T>::value
			       , ClassGetNext<T>
			       , TypeGetNext<T>
			       >::type {};

  /*********************************************/
  /*  ___    _    _ _                          */
  /* | __|__| |__| | |   __ _ _  _ ___ _ _ ___ */
  /* | _/ _ \ / _` | |__/ _` | || / -_) '_(_-< */
  /* |_|\___/_\__,_|____\__,_|\_, \___|_| /__/ */
  /*                          |__/             */
  /*********************************************/
  template<class LayerFn, class NetType>
  struct FoldLayers {
    template<class ... Aux>
    static void map(Aux&& ... aux) {
      LayerFn::apply(aux...);
      /* Majestic... */
      FoldLayers<LayerFn, typename NetType::Next
		 >::map( GetNext<typename std::remove_reference<Aux>::type
			 >::apply( std::forward<typename std::remove_reference<Aux>::type >(aux) ) ...);
    }
  };

  template<class LayerFn>
  struct FoldLayers<LayerFn,_void> {
    template<class ... Aux> static void map(Aux...) { }
  };

  /*************************************************/
  /*  ___ ___    _    _ _                          */
  /* | _ \ __|__| |__| | |   __ _ _  _ ___ _ _ ___ */
  /* |   / _/ _ \ / _` | |__/ _` | || / -_) '_(_-< */
  /* |_|_\_|\___/_\__,_|____\__,_|\_, \___|_| /__/ */
  /*                              |__/             */
  /*************************************************/
  template<class LayerFn, class NetType>
  struct RFoldLayers {
    template<class ... Aux>
    static void map(Aux&& ... aux) {
      RFoldLayers<LayerFn, typename NetType::Next
		  >::map( GetNext<typename std::remove_reference<Aux>::type
			  >::apply( std::forward<typename std::remove_reference<Aux>::type >(aux) ) ...);
      LayerFn::apply(aux...);
    }};

    template<class LayerFn>
    struct RFoldLayers<LayerFn,_void> {
      template<class ... Aux> static void map(Aux...) { }
    };

  /******************************************************************/
  /*  ___    _    _ ___                __      __   _      _   _    */
  /* | __|__| |__| | __|_ _____ _ _ _  \ \    / /__(_)__ _| |_| |_  */
  /* | _/ _ \ / _` | _|\ V / -_) '_| || \ \/\/ / -_) / _` | ' \  _| */
  /* |_|\___/_\__,_|___|\_/\___|_|  \_, |\_/\_/\___|_\__, |_||_\__| */
  /*                                |__/             |___/          */
  /******************************************************************/
  struct FoldEveryWeight {
    template<class Fn, class LayerFn, class Net>
    static void apply(Fn fn, LayerFn layer_fn, Net net) {
      layer_fn();
      array_fold( [&](decltype(net.layer[0])&& weights) {
	  array_fold(fn, weights);
	}, net.layer);
    }

    template<class Fn, class NodeFn, class LayerFn, class Net>
    static void apply(Fn fn, NodeFn node_fn, LayerFn layer_fn, Net net) {
      layer_fn();
      array_fold( [&](decltype(net.layer[0])&& weights) {
	  node_fn();
	  array_fold(fn, weights);
	}, net.layer);
    }

    template<class Fn, class Net>
    static void apply(Fn fn, Net net) {
      array_fold( [&](decltype(net.layer[0])&& weights) {
	  array_fold(fn, weights);
	}, net.layer);
    }};


  struct MapFeed {
    template<class Fn, class Feed, class ... Rest>
    static void apply(Fn fn, Feed &feed, Rest& ... rest) {
      ref_map(fn, feed.layer, rest.layer...);
    }};

  struct MapFeedLayers {
    template<class Fn, class LayerFn, class Feed>
    static void apply(Fn fn, LayerFn layer_fn, Feed &feed) {
      layer_fn();
      array_fold(fn, feed.layer);
    }};

  /*********************************************************************/
  /*  __  __           ___                __      __   _      _   _    */
  /* |  \/  |__ _ _ __| __|_ _____ _ _ _  \ \    / /__(_)__ _| |_| |_  */
  /* | |\/| / _` | '_ \ _|\ V / -_) '_| || \ \/\/ / -_) / _` | ' \  _| */
  /* |_|  |_\__,_| .__/___|\_/\___|_|  \_, |\_/\_/\___|_\__, |_||_\__| */
  /*             |_|                   |__/             |___/          */
  /*********************************************************************/
  struct MapWeight {
    template<class Fn, class ... Nets>
    static void apply(Fn fn, Nets& ... rest) {
      ::ref_map( [&](decltype(rest.layer[0])& ... weights) {
	  ::ref_map(fn, weights...);
	}, rest.layer ...);
    }};

  struct MapWeightLayer {
    template<class Fn, class LayerFn, class ... Nets>
    static void apply(Fn fn, LayerFn layer_fn, Nets& ... net) {
      layer_fn();
      array_fold( [&](decltype(net.layer[0])& ... weights) {
	  ::ref_map(fn, weights...);
	}, net.layer...);
    }};
}

/*********************************************/
/*  ___                                      */
/* | _ \___ __ _  _ _ _ _ _ ___ _ _  __ ___  */
/* |   / -_) _| || | '_| '_/ -_) ' \/ _/ -_) */
/* |_|_\___\__|\_,_|_| |_| \___|_||_\__\___| */
/*********************************************/
template<class Fn, class Net>
void fold(const Fn fn, const Net& net) {
  using namespace recurrence_detail;
  FoldLayers<FoldEveryWeight, Net>::map(fn, const_cast<Net&>(net));
}

template<class Fn, class LayerFn, class Net>
void fold(Fn fn, LayerFn layer_fn, Net& net) {
  using namespace recurrence_detail;
  FoldLayers<FoldEveryWeight, Net>::map(fn, layer_fn, net);
}

template<class Fn, class NodeFn, class LayerFn, class Net>
void fold(Fn fn, NodeFn node_fn, LayerFn layer_fn, Net& net) {
  using namespace recurrence_detail;
  FoldLayers<FoldEveryWeight, Net>::map(fn, node_fn, layer_fn, net);
}

template<class Fn, class Net, class ... Rest>
void map(Fn fn, Net& net, Rest& ... rest) {
  using namespace recurrence_detail;
  FoldLayers<MapWeight, Net>::map(fn, net, rest...);
}



/**
 * Prints the networks weights for each layer
 * @tparam Net:
 * @return: 
 */
template<class Net>
std::ostream& print_network(Net& net, std::ostream &out) {
  using namespace std;
  static_assert( std::is_same<NNet_tag, typename Net::tag>::value
		 , "print_network only works for NNet");

  out << "(l";
  fold([&](float f) { out << " " << f; }
      , [&]() { out << ") (n"; }
      , [&]() { out << ")\n(l"; }
      , net);
  out << ")";

  return out;
}

template<class Net>
std::ostream& print_network(Net& net) {  return print_network(net, std::cout); }

template<class Feed>
std::ostream& print_feed(Feed& feed, std::ostream &out) {
  using namespace recurrence_detail;
  using namespace std;

  static_assert( is_same<FeedNet_tag, typename Feed::tag>::value
		 , "print_feed only works for a FeedNet");

  FoldLayers<MapFeedLayers, Feed
	     >::map([&](float ff)
		    { cout << " " << ff; }
		    , [&] () { cout << endl; }
		    , feed);
  return out;
}

template<class Feed>
std::ostream& print_feed(Feed& feed) { return print_feed(feed, std::cout); }


template<class Net>
std::istream& read_layer(Net& net, std::istream &in) {
  using namespace std;
  net.map([&](float& f) {
      char c = in.peek();
      while( !(c == '-' || (c > '0' && c < '9')) ) {
	in.ignore();
	c = in.peek();
      }
      in >> f;
    });
  return in;
}

template<class Net>
void permute(Net &n, float low, float high) {
  using namespace std;
  using namespace recurrence_detail;

  random_device rd;
  mt19937 gen(rd());

  uniform_real_distribution<float> dist(low, high);

  FoldLayers<MapWeight, Net
	     >::map([&](float& f)  {
		 f += dist(gen);
	       }, n);
}

/*******************************/
/*  ___            _ _    _    */
/* | _ \_ _ ___ __| (_)__| |_  */
/* |  _/ '_/ -_) _` | / _|  _| */
/* |_| |_| \___\__,_|_\__|\__| */
/*******************************/
namespace recurrence_detail {
  /* compiler kevetches about local structs and templates, otherwise I'd put this in
     predict */
  struct PredictMap {
    template<class Net>
    static void apply(Net& net, typename Net::Feed& feed) {
      using namespace std;
      static_assert( tuple_size<typename Net::NodeInput>::value == tuple_size<typename Net::Feed::Layer>::value + 1
		     , "PredictMap feed/net dimention mismatch.");
      ref_map([&](decltype(net.layer[0])& node
		  , float &dst) {
		ref_map( [&](float &nn, float &ff) {
		    /* map over all the non-bias inputs */
		    dst += nn * ff;
		  }, node
		  , feed.layer);
		/* include the bias input */
		dst = sigmoid(dst + node.back());
	      } , net.layer
	      , feed.next.layer);
    }};
}

template<class Net>
typename Net::Feed& predict(Net& net, typename Net::Feed& feeds) {
  using namespace recurrence_detail;
  FoldLayers<PredictMap, Net>::map(net, feeds);

  return feeds;
}

#endif
