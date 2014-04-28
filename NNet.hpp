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

#include "./utility.hpp"

template<class T>
struct GetOutput { typedef typename T::Output type; };

template<class T>
struct GetInput { typedef typename T::Input type; };


struct TagBase {};
/****************************************/
/*  _____             _ _   _      _    */
/* |  ___|__  ___  __| | \ | | ___| |_  */
/* | |_ / _ \/ _ \/ _` |  \| |/ _ \ __| */
/* |  _|  __/  __/ (_| | |\  |  __/ |_  */
/* |_|  \___|\___|\__,_|_| \_|\___|\__| */
/****************************************/
struct FeedNet_tag : public TagBase { typedef FeedNet_tag tag; };

namespace detail {
  /* probably a clever way to do this */
  template<class Feed, class Next = typename Feed::Next>
  struct GetOutputLayer {
    static typename Next::Output& apply(Next &next, typename Feed::Layer &net) { return next.output_layer(); }};

  template<class Feed>
  struct GetOutputLayer<Feed,_void> {
    static typename Feed::Output& apply(_void , typename Feed::Layer &layer) { return layer; }};
}

template<class NN, class Numeric = float>
struct FeedNet : public FeedNet_tag {
  typedef Numeric Num;

  typedef FeedNet<NN> type;
  typedef typename FeedNet< typename NN::Next >::type Next;
  typedef std::array<Num, NN::value > Layer;

  typedef NN Dimention;
  const static size_t depth = NN::depth;
  

  typedef typename std::conditional< std::is_same<_void, Next>::value
				     , Identity< Layer >
				     , GetOutput< Next >
				     >::type::type Output;

  Layer layer;
  Next next;
  
  Output& output_layer() {
    return detail::GetOutputLayer<FeedNet, Next>::apply(next, layer);
  }

  FeedNet() : layer{} {}
  FeedNet(Num n) : next(n) { map_array([&](Num &f) { f = n; }, layer); }
  FeedNet(const FeedNet &feed) : layer(feed.layer), next(feed.next) {}
};
  
/* Base class */
template<class Numeric>
struct FeedNet< _void, Numeric> : public _void {
  typedef Numeric Num;
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
 * @tparam Layers: an array of std::array<Numeric, size>
 */
struct NNet_tag : public TagBase { typedef NNet_tag tag; };

template<class GenNN, class Numeric = float>
class NNet : public NNet_tag {
  typedef typename GenNN::type NN;
public:
  typedef Numeric Num;
  typedef NNet<NN> type;
  typedef FeedNet< NN > Feed;

  static const size_t depth = NN::depth;

  typedef typename NNet<typename NN::Next>::type Next;

  typedef typename std::conditional< IsVoid<Next>::value
				     , Identity< std::array<Num, NN::value> >
				     , GetInput<Next>
				     >::type::type LayerOutput;

  typedef typename std::conditional< IsVoid<Next>::value
				     , Identity< Identity< LayerOutput > >
				     , std::enable_if<!IsVoid<Next>::value, GetOutput<Next> >
				     >::type::type::type Output;

  typedef std::array<Num, NN::value> Input;

  typedef std::array<Num, NN::value + 1> NodeInput;
  typedef std::array<NodeInput, std::tuple_size< LayerOutput >::value > Layer;

  Next next;
  Layer layer;
  
  template<class ... Rest>
  NNet(const Layer& init, Rest ... rest) :  next(rest ...) , layer(init) {}

  NNet(const NNet& net) : next(net.next), layer(net.layer) {}
  NNet() : layer{} {}
};

template<class Numeric> class NNet< _void, Numeric > : public _void {
  typedef Numeric Num;
};

template<class T>
struct NumType : public Identity<typename T::Num> {};

/*************************************************/
/*  _____                 _   _                  */
/* |  ___|   _ _ __   ___| |_(_) ___  _ __  ___  */
/* | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __| */
/* |  _|| |_| | | | | (__| |_| | (_) | | | \__ \ */
/* |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/ */
/*************************************************/
namespace recurrence_detail {
  template<class T>
  struct PassThrough {
    typedef T Next;
    typedef typename std::remove_reference<T>::type Bare;

    template<class U> 		/* needs to be template so it'll collapse the && when needed */
    static U&& apply(U&& input) {
      return std::forward<U>(input);
    }};

  template<class T>
  struct ClassGetNext {
    typedef typename std::remove_reference<T>::type Bare;
    typedef typename Bare::Next Next;

    template<class U> 		/* needs to be template so it'll collapse the && when needed */
    static auto apply(U&& input) -> decltype(input.next)&& {
      return std::forward<Next>( input.next );
    }};

  template<class T>
  struct is_net_or_fee {
    typedef typename std::remove_reference<T>::type Bare;
    static constexpr bool value = std::is_class<Bare>::value && std::is_base_of<TagBase, Bare>::value;

    typedef is_net_or_fee<Bare> type;
  };

  template<class T>
  struct GetNext
    : public std::conditional< is_net_or_fee<T>::value
			       , ClassGetNext<T>
			       , PassThrough<T>
			       >::type {};

  /************************************************/
  /*  __  __           _                          */
  /* |  \/  |__ _ _ __| |   __ _ _  _ ___ _ _ ___ */
  /* | |\/| / _` | '_ \ |__/ _` | || / -_) '_(_-< */
  /* |_|  |_\__,_| .__/____\__,_|\_, \___|_| /__/ */
  /*             |_|             |__/             */
  /************************************************/
  /**
   * Takes a struct which applies a templated functin to each layer of some input NNet[s] and/or Feed[s]
   *
   * Default behavior is to apply to each row _expect for the last_ so that the function applied by LayerFn can
   * use layer.next and get valid results.
   * 
   * Use the Offset template parameter to change this behavious; 1 to iterate over the last layer, -1 to leave off the last two layers, -2 to leave off the last three etc.
   */
  template<class LayerFn, class NetType, long int Offset = 0, size_t D = NetType::depth + Offset>
    struct MapLayers {
      template<class ... Aux>
      static void map(Aux&& ... aux) {
	LayerFn::apply(aux...);
	/* Majestic... */
	MapLayers<LayerFn, typename NetType::Next, 0, D - 1
		  >::map( GetNext<typename std::remove_reference<Aux>::type
			  >::apply( std::forward<typename std::remove_reference<Aux>::type >(aux) ) ...);
      }
    };

  template<class LayerFn, class T, long int Offset>
  struct MapLayers<LayerFn, T, Offset, 0> {
    template<class ... Aux> static void map(Aux...) { }
  };

  /****************************************************/
  /*  ___ __  __           _                          */
  /* | _ \  \/  |__ _ _ __| |   __ _ _  _ ___ _ _ ___ */
  /* |   / |\/| / _` | '_ \ |__/ _` | || / -_) '_(_-< */
  /* |_|_\_|  |_\__,_| .__/____\__,_|\_, \___|_| /__/ */
  /*                 |_|             |__/             */
  /****************************************************/
  template<class LayerFn, class NetType, size_t Offset = 0, size_t D = NetType::depth - Offset>
    struct RMapLayers {
      template<class ... Aux>
      static void map(Aux&& ... aux) {
	RMapLayers<LayerFn, typename NetType::Next, 0, D - 1
		   >::map( GetNext<typename std::remove_reference<Aux>::type
			   >::apply( std::forward<typename std::remove_reference<Aux>::type >(aux) ) ...);
	LayerFn::apply(aux...);
      }};

  template<class LayerFn, class T, size_t Offset>
  struct RMapLayers<LayerFn, T, Offset, 0> {
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
      map_array(fn, feed.layer, rest.layer...);
    }};

  struct MapFeedLayers {
    template<class Fn, class LayerFn, class Feed>
    static void apply(Fn fn, LayerFn layer_fn, Feed &feed) {
      layer_fn();
      map_array(fn, feed.layer);
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
      map_array( [&](decltype(rest.layer[0])& ... weights) {
	  map_array(fn, weights...);
	}, rest.layer ...);
    }};

  struct MapWeightLayer {
    template<class Fn, class LayerFn, class ... Nets>
    static void apply(Fn fn, LayerFn layer_fn, Nets& ... net) {
      layer_fn();
      array_fold( [&](decltype(net.layer[0])& ... weights) {
	  ::map_array(fn, weights...);
	}, net.layer...);
    }};
}

/*********************************************/
/*  ___                                      */
/* | _ \___ __ _  _ _ _ _ _ ___ _ _  __ ___  */
/* |   / -_) _| || | '_| '_/ -_) ' \/ _/ -_) */
/* |_|_\___\__|\_,_|_| |_| \___|_||_\__\___| */
/*********************************************/
template<class Fn, class Net, class ... Rest>
void map_network(Fn fn, Net& net, Rest& ... rest) {
  using namespace recurrence_detail;
  MapLayers<MapWeight, Net>::map(fn, net, rest...); }

template<class Fn, class Feed, class ... Rest>
void map_feed(Fn fn, Feed &feed, Rest& ... rest) {
  using namespace recurrence_detail;
  MapLayers<MapFeed, Feed>::map(fn, feed, rest...); }

/**
 * Prints the networks weights for each layer
 * @tparam Net:
 * @return: 
 */
namespace recurrence_detail {
  struct PrintNet {
    template<class Net>
    static void apply(Net &net, std::ostream &out) {
      using namespace std;
      map_array( [&](typename Net::NodeInput& node) {
	  cout << "(" << node[0];
	  Map<1,0>::apply([&](const typename NumType<Net>::type nn) {
	      out << ", " << nn;
	    }, node);
	  cout << ")";
	}, net.layer);
      cout << "\n";
    }}; }

template<class Net>
std::ostream& print_network(Net& net, std::ostream &out) {
  using namespace recurrence_detail;
  using namespace std;

  static_assert( std::is_same<NNet_tag, typename Net::tag>::value
		 , "print_network only works for NNet");

  MapLayers<PrintNet, Net>::map(net, ref(out) );
  return out;
}

template<class Net>
std::ostream& print_network(Net& net) {
  return print_network(net, std::cout);
}

template<class Feed>
std::ostream& print_feed(Feed& feed, std::ostream &out) {
  using namespace recurrence_detail;
  using namespace std;

  static_assert( is_same<FeedNet_tag, typename Feed::tag>::value
		 , "print_feed only works for a FeedNet");

  MapLayers<MapFeedLayers, Feed, 1
	    >::map([&](NumType<Feed> ff)
		    { cout << " " << ff; }
		    , [&] () { cout << endl; }
		    , feed);
  return out;
}
template<class Feed>
std::ostream& print_feed(Feed& feed) { return print_feed(feed, std::cout); }


/* dead simple, assumes the input is a white space seperated list of node weights.
   No dimention checking or anything is done, just tries to slurp them in. */
template<class Net>
std::istream& read_net(Net& net, std::istream &in) {
  using namespace std;
  map_network([&](typename NumType<Net>::type& f) {
      in >> f;
    }, net);
  return in;
}

/* dead simple output, puts each node out onto a stream as a float.
   Should be compatible with read_net. */
template<class Net>
std::ostream& write_net(Net& net, std::ostream &out) {
  map([&](typename NumType<Net>::type& f) { out << f << " "; }, net);
  return out;
}

template<class Net>
void permute(Net &n, float low, float high) {
  using namespace std;
  using namespace recurrence_detail;
  typedef typename NumType<Net>::type Num;

  random_device rd;
  mt19937 gen(rd());

  uniform_real_distribution<> dist(low, high);

  MapLayers<MapWeight, Net
	     >::map([&](Num& f)  {
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
      typedef typename NumType<Net>::type Num;

      static_assert( tuple_size<typename Net::NodeInput>::value == tuple_size<typename Net::Feed::Layer>::value + 1
		     , "PredictMap feed/net dimention mismatch.");
      map_array([&](typename Net::NodeInput& node
		  , Num &dst) {
		dst = 0.0;
		map_array( [&](Num &nn, Num &ff) {
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
  MapLayers<PredictMap, Net>::map(net, feeds);

  return feeds;
}

/*************************************************/
/*     _                                    _    */
/*    / \  _   _  __ _ _ __ ___   ___ _ __ | |_  */
/*   / _ \| | | |/ _` | '_ ` _ \ / _ \ '_ \| __| */
/*  / ___ \ |_| | (_| | | | | | |  __/ | | | |_  */
/* /_/   \_\__,_|\__, |_| |_| |_|\___|_| |_|\__| */
/*               |___/                           */
/*************************************************/
/* add a bias value to a feed vector */
namespace recurrence_detail {
  /* I don't want to increment the output layer, just the input and hidden. */
  /* template<class NNs, class Rest> */
  /* struct IncrementNums { */
  /*   typedef ConsNums<NNs::value + 1, typename IncrementNums<Rest, typename Rest::Next>::type > type; */
  /* }; */

  /* template<class NNs> */
  /* struct IncrementNums<NNs, _void> : public NNs {}; */
  template<class NNs>
  struct IncrementNums {
    typedef ConsNums<NNs::value + 1, typename IncrementNums<typename NNs::Next>::type > type;
  };

  template<>
  struct IncrementNums<_void> : public _void {};

  struct AugCopy {
    template<class LL_in, class LL_out>
    static void apply(LL_in& in, LL_out& out) {
      typedef typename NumType<LL_in>::type Num;

      map_array([](Num &in_v, Num &out_v) { out_v = in_v; }, in.layer, out.layer);
      out.layer.back() = 1.0;
    }};
}

template<class Feed>
struct Augment {
  typedef FeedNet<typename recurrence_detail::IncrementNums< typename  Feed::Dimention >::type
		  > type;
  
  static type apply(Feed &input) {
    using namespace recurrence_detail;
    type augmented;
    MapLayers<AugCopy, Feed, 1>::map(input, augmented);
    return augmented;
  }

  static void apply(Feed &src, type &dst) {
    using namespace recurrence_detail;
    MapLayers<AugCopy, Feed, 1>::map(src, dst);
  }
};

#endif
