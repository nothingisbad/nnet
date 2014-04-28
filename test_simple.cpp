/**
 * @file /home/ryan/programming/nnet/test_simple.cpp
 * @author Ryan Domigan <ryan_domigan@sutdents@uml.edu>
 * Created on Apr 14, 2014
 *
 * A simple forward progigation test.  I'm comparing against 'known good' values
 * from a matlab based neural network from my machine learning class.
 */

#include "NNet.hpp"
#include "gradient_decent.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

int main() {
  using namespace std;
  using namespace recurrence_detail;

  typedef NNet< Nums<1, 2, 2> > Net;

  /* Specify the networks initial weights */
  Net net( array< array<float,2>, 2>{{ array<float,2>{{0.1, 0.2}}, array<float,2>{{0.3, 0.4}}}},

	       array< array<float,3>, 2>{{ array<float,3>{{0.5, 0.6, 0.7}}
		                          , array<float,3>{{0.8, 0.9, 1.0}} } });


  typedef typename Net::Feed Feed;
  Feed feed;

  feed.layer = array<float,1>{{0}};

  auto error = [](float hyp, float lable) -> bool { return lable > 0.5 ? hyp > 0.5 : hyp < 0.5; };

  auto mk_input = [](float ff) -> array<float,1> { return array<float,1>{{ff}}; };
  vector< typename Net::Feed::Layer > X;
  vector< typename Net::Output > Y;
  X.push_back(mk_input(0));		/* input 1*/
  Y.push_back(array<float,2>{{1.0, 0.0}});

  cout << "Training label: ";
  print_array(Y[0]) << endl;

  /* try the forward feed */
  auto octave_result = array<float, 2>{{0.79152, 0.87854}};
  predict(net, feed);

  cout << "Predicted:\n";
  map_array([&](float &aa, float& bb ) {
      cout << "  " << aa << " (" << bb << ")\n";
    }, feed.output_layer(), octave_result);

  cout << "Prediction layers: \n";
  print_feed(feed) << endl;
  
  /* get the cost/gradient */
  auto cost = cost_function(net, X, Y, error, 1);

  cout << "Cost: " << get<1>(cost) << " (" << 3.4220 << ")\n";

  cout << "Network: \n";
  print_network(net, cout) << "\n";

  cout << "Gradient: " << endl;

  /* print out the inputs and corrosponding gradient I've just computed, with the 'known good' values in the next column to their right  */
  auto octave_gradient = array<float, 10>{{0.10000, 0.14816, 0.30000, 0.15992, 0.38537, 0.47519
					   , -0.2084, 1.28305, 1.42597, 0.87854}};

  int i = 0;
  map_network([&](float input, float mine) {
      cout << setw(5) << input << " -> "<<  setw(10) << mine << "  " << octave_gradient[i++] << endl;
    }, net, get<0>(cost) );

  cout << "Output network: \n";
  map_network([](float &net, float &grad) { net += grad; }, net, get<0>(cost));
  print_network(net, cout) << "\n";

  return 0;
}

