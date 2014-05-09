
/**
 * @file /home/ryan/programming/nnet/test/test_regression.cpp
 * @author Ryan Domigan <ryan_domigan@sutdents@uml.edu>
 * Created on May 08, 2014
 *
 * Test the regressive abilities of the gradient decent learner
 * Similar to http://www.mathworks.com/help/nnet/ug/improve-neural-network-generalization-and-avoid-overfitting.html
 */

#include <nnet/NNet.hpp>
#include <nnet/gradient_decent.hpp>
#include <nnet/utility.hpp>

#include <fstream>
#include <math.h>

using namespace nnet;
using namespace std;

const static size_t num_bins = 20;
typedef NNet< Nums<1, 20, num_bins> > Net;
typedef typename Net::Feed Feed;

/* Discretize the output so I can use the sigmoid activation function */
typename Feed::Output float2label(float f) {
  Feed::Output ff{};
  int label = (f + 1) * (num_bins / 2);
  if(label >= (int)num_bins) label = (num_bins - 1);
  if(label < 0) label = 0;

  ff[(int) ((f + 1) * 10) ] = 1;;
  return ff;
}

/* put me back in a -1 to 1 range */
float label2float(const Feed::Output& layer) {
  int i = max_element(layer.begin(), layer.end()) - layer.begin();
  cout << "Index: " << i << " yielding " << ((i * 2) / (float)num_bins) - 1 << endl;
  return (i * 2) / num_bins - 1;
}


int main() {
  Net net{};
  Feed feed;
  vector< Feed::Layer > X;
  vector< Feed::Output > Y;

  initialize(net, -0.12, 0.12);;

  {
    ifstream file("./noisy-sine.txt");
    float x, y;

    while( file >> x && file >> y) {
      X.push_back( Feed::Layer{{x}} );
      Y.push_back( float2label( y ) );
    }}

  /* for(int i = 0; i < 10; ++i) { */
  /*   cout << "X:"; */
  /*   print_array(X[i]); */
  /*   cout << " Y:"; */
  /*   print_array(Y[i]) << endl; */
  /* } */

  for(size_t i = 0; i < 4000; ++i)
    train(net, X, Y, 0.01);

  {
    ofstream file("learned-sin.txt");
    for(float x = - 2 * M_PI
	  ; x < (2 * M_PI)
	  ; x += (M_PI / 4)) {
      feed.layer[0] = x;
      // print_array(feed.output_layer()) << endl;

      predict(net,feed);
      label2float( feed.output_layer() );

      file << x << " " << label2float( feed.output_layer() ) << endl;
    }}

  cout << "Done" << endl;

  return 0;
}
