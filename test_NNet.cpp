/**
 * @file /home/ryan/programming/nnet/test_NNet.cpp
 * @author Ryan Domigan <ryan_domigan@sutdents@uml.edu>
 * Created on Apr 06, 2014
 */

#include "NNet.hpp"

#include <iostream>


int main() {
  using namespace std;
  using namespace recurrence_detail;

  typedef NNet< array<float,2>
  		, array<float,4>
  		, array<float,3>
  		, array<float,1> >
    NetType;

  NetType net;
  typedef typename NetType::Feed Feed;
  Feed feed;

  feed.layer = array<float,2>{{1,1}};

  int row = 0,
    i = 0;
  fold([&](float) { ++i; }
    , [&]() { cout << "Row " << row <<" has " << i << " elements " << endl;
      ++row;
      i = 0;
    }
    , net);

  print_network(net, cout) << "\n\n";

  permute(net, -0.12, 0.12);

  print_network(net, cout) << "\n\n";

  predict(net, feed);

  cout << " (Forward fed: ";
  MapLayers<MapFeedLayers, Feed
	     >::apply( [&](float f) {
		 cout << f << " ";
	       } , [&]() {
		 cout << endl;
	       } , feed);

  cout << ")" << endl;

  i = 0;
  back_propigate( [&](float theta, float err, float &dst) {
      dst = i++;
    }, net, feed);

  cout << " (back propigated: ";
  MapLayers<MapFeedLayers, Feed
	     >::apply( [&](float f) {
		 cout << f << " ";
	       } , [&]() {
		 cout << "\n  ";
	       } , feed);
  cout << ")" << endl;



  return 0;
}
o
