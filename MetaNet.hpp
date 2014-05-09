#ifndef METANET_HPP
#define METANET_HPP
/**
 * @file /home/ryan/programming/nnet/MetaNet.hpp
 * @author Ryan Domigan <ryan_domigan@sutdents@uml.edu>
 * Created on Apr 28, 2014
 *
 * Meta net nodes are themselves fully interconnected feed-forwards neural nets, but
 * nodes cannot see their other nodes of their layer.
 */

#include "./NNet.hpp"
#include "./utility"

namespace nnet {
  namespace recurrence_detail {
    struct SumOutputs {
      template<class Net, class Accume>
      struct Apply : public intC< Net::output_size + Accume::type::value > {};
    };

    struct SumInputs {
      template<class Net, class Accume>
      struct Apply : public intC< Net::input_size + Accume::type::value > {};
    }; 

    template<class List>  
    struct Net2FeedList
      : public Cons< typename Car<List>::type::Feed
		     , Net2FeedList<typename Cdr<List>::type >
		     >::type {};
    template<>
    struct Net2FeedList<_void> : public _void {};
  }

  /**********************************************/
  /*  __  __      _        _____             _  */
  /* |  \/  | ___| |_ __ _|  ___|__  ___  __| | */
  /* | |\/| |/ _ \ __/ _` | |_ / _ \/ _ \/ _` | */
  /* | |  | |  __/ || (_| |  _|  __/  __/ (_| | */
  /* |_|  |_|\___|\__\__,_|_|  \___|\___|\__,_| */
  /**********************************************/

  struct MetaFeed_tag : public TagBase { typedef MetaFeed_tag tag; };

  template<class NetList>
  struct MetaFeed : public MetaFeed_tag {
    typedef MetaFeed<NetList> type;
    typedef typename Car<NetList>::type::Num Num;
    typedef typename recurrence_detail::Net2FeedList<NetList>::type FeedList;

    static const size_t layer_output_size
    = FoldMPList< recurrence_detail::SumOutputs, intC<0>, NetList >::type::value; 

    static const size_t layer_input_size
    = FoldMPList< recurrence_detail::SumInputs, intC<0>, NetList
		  >::type::value;

    static const size_t total_input_size = layer_input_size;
    static const size_t total_output_size = layer_output_size;

    static const size_t node_count = NetList::depth;

    FeedList instance_list;
  };

  /*****************************************/
  /*  __  __      _        _   _      _    */
  /* |  \/  | ___| |_ __ _| \ | | ___| |_  */
  /* | |\/| |/ _ \ __/ _` |  \| |/ _ \ __| */
  /* | |  | |  __/ || (_| | |\  |  __/ |_  */
  /* |_|  |_|\___|\__\__,_|_| \_|\___|\__| */
  /*****************************************/

  struct MetaNet_tag : public TagBase { typedef MetaNet_tag tag; };

  template<class NetList>
  class MetaNet : public MetaNet_tag {
  public:
    typedef MetaNet<NetList> type;

    typedef typename Car<NetList>::type::Num Num;
    typedef MetaFeed<NetList> Feed;

    typedef std::array<Num , MetaFeed<NetList>::total_input_size
		       > Input;

    NetList instance_list;
  };

  template<> class MetaNet<_void> : public _void {};


  /*************************************************/
  /*  _____                 _   _                  */
  /* |  ___|   _ _ __   ___| |_(_) ___  _ __  ___  */
  /* | |_ | | | | '_ \ / __| __| |/ _ \| '_ \/ __| */
  /* |  _|| |_| | | | | (__| |_| | (_) | | | \__ \ */
  /* |_|   \__,_|_| |_|\___|\__|_|\___/|_| |_|___/ */
  /*************************************************/
  namespace recurrence_detail {
    template<class Fn, class List>
    struct MetaMap {
      template<class ... Aux>
      static void apply(Aux&& ... aux) {
	MapLayers<Fn, typename Car<List>::type>::apply(aux ...);

	MetaMap<Fn, typename Car<List>::type >::apply( GetNext<Aux>::apply( std::forward<Aux>(aux) ) ...);
      }};

    template<class Fn> struct MetaMap<Fn, _void> {
      template<class ... Aux> void apply(Aux &&...) {}
    };

    struct MapToInput {
      template<class Fn, class Feed>
      void apply(Fn fn, Feed &feed) { map_array(fn, feed.layer); }};

    struct MapToOutput {
      template<class Fn, class Feed>
      void apply(Fn fn, Feed &feed) { map_array(fn, feed.output_layer()); }};

    struct WrapPredict { template<class Net> void apply(Net& net, typename Net::Feed& feed) { predict(net, feed); } };
  }

  /**
   * Input a network and a feed (with the input values entered), feeds inputs forwards
   * through feed, storing the results in the output layer.
   * 
   * @tparam MNet:
   * @param net:
   * @param feed:
   */
  template<class MNet>
  void predict_meta(MNet& net, typename MNet::Feed& feed) {
    using namespace recurrence_detail;
    typedef typename NumType<MNet>::type Num;

    MetaMap<WrapPredict, MNet>::apply(net, feed);
  }
}
#endif
