
/**
 * @file /home/ryan/programming/nnet/test_MetaNet.cpp
 * @author Ryan Domigan <ryan_domigan@sutdents@uml.edu>
 * Created on Apr 28, 2014
 *
 * Similar to test_1d, but using a different topology
 */

#include "MetaNet.hpp"
#include "gradient_decent.hpp"

#include <iostream>
#include <fstream>
#include <tuple>
#include <algorithm>

using namespace std;

int main() {
  typedef
    typename MetaNet< MPList< MPList< NNet< Nums<2, 1> > , NNet< Nums<2,1> > >
			      , MPList< NNet< Nums<3, 1> > > >
		      >::type
    MNet;
  typedef typename MNet::Feed Feed;
  
  cout << "Inputs: " << Feed::total_input_size  << "Outputs: " << Feed::total_output_size << endl;
  
  return 0;
}
