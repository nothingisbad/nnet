/**
 * @file /home/ryan/programming/nnet/test_NNet.cpp
 * @author Ryan Domigan <ryan_domigan@sutdents@uml.edu>
 * Created on Apr 06, 2014
 */

#include "NNet.hpp"
#include "./gradient_decent.hpp"

#include <iostream>


int main() {
  using namespace std;

  typedef NNet< array<float,2>
  		, array<float,4>
  		, array<float,3>
  		, array<float,1> >
    NetType;

  NetType net;

  int row = 0,
    i = 0;
  fold([&](float) {
      ++i; }
    , [&]() { cout << "Row " << row <<" has " << i << " elements " << endl; }
    , net);

  print_theta(net, cout) << "\n\n";


  return 0;
}
