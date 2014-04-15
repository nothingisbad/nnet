/**
 * @file /home/ryan/programming/nnet/test_simple.cpp
 * @author Ryan Domigan <ryan_domigan@sutdents@uml.edu>
 * Created on Apr 14, 2014
 */

#include "NNet.hpp"
#include "gradient_decent.hpp"
#include <iostream>
#include <vector>

int main() {
  using namespace std;
  using namespace recurrence_detail;

  cout << "Sigmoid: " << sigmoid(0.4) << " " << sigmoid(0.8) << " " << sigmoid(2 * 0.6 * 0.4 + 0.4) << endl;

  typedef NNet< array<float,1>
  		, array<float,2>
  		, array<float,2> >
    Net;

  Net net( array< array<float,2>, 2>{{ array<float,2>{{0.4, 0.4}}, array<float,2>{{0.4, 0.4}}}},

	       array< array<float,3>, 2>{{ array<float,3>{{0.4, 0.4, 0.4}}
		                          , array<float,3>{{0.4, 0.4, 0.4}} } });

  typedef typename Net::Feed Feed;
  Feed feed;

  feed.layer = array<float,1>{{0}};

  auto error = [](float _, float __) -> bool { return false; };

  auto mk_input = [](float ff) -> array<float,1> { return array<float,1>{{ff}}; };
  vector< typename Net::Feed::Layer > X;
  vector< typename Net::Output > Y;
  X.push_back(mk_input(0));		/* input 1*/
  Y.push_back(array<float,2>{{1.0, 1.0}});

  cout << "Training label: ";
  print_array(Y[0]) << endl;
  
  auto cost = cost_function(net, X, Y, error, 2);
  cout << "Cost: " << get<1>(cost) << "\n";

  cout << "Gradient: " << endl;
  print_network(get<0>(cost), cout) << "\n";

  map([](float &net, float &grad) { net += grad; }, net, get<0>(cost));

  print_network(net, cout) << "\n";

  return 0;
}

