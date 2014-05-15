
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

#define CONTINUOUS 1

#ifdef CONTINUOUS
const static size_t num_bins = 1;
typedef NNet< Nums<1, 12, num_bins>
	      , double
	      , Gaussian<double, rational_c<3,2> >
	      > Net;
typedef typename Net::Feed Feed;

typename Feed::Output float2label(float f) {
  Feed::Output ff{};
  /* translate outputs into (1,0)  */
  ff[0] = (f + 1.1) / 2.2;

  if(ff[0] >= 1) ff[0] = 1;
  else if(ff[0] < 0) ff[0] = 0;

  return ff;
}

/* put me back in a -1 to 1 +/- noise range */
float label2float(const Feed::Output& layer) {
  typedef typename Net::Num RealNum;
  RealNum ll = (RealNum)layer[0];
  return (ll * 2.2) - 1.1;
}

#else  /**** <<< CONTINUOUS
	     >>> DISCRETE ****/

const static size_t num_bins = 20;
typedef NNet< Nums<1, 20, num_bins>
	      // , double
	      // , Gaussian<double, rational_c<3,2> >
	      > Net;
typedef typename Net::Feed Feed;


/* Discretize the output so I can use the sigmoid activation function */
typename Feed::Output float2label(float f) {
  typedef typename Net::Num RealNum;
  Feed::Output ff{};
  int label = (f + 1) * (num_bins / 2);
  if(label >= (int)num_bins) label = (num_bins - 1);
  if(label < 0) label = 0;

  (RealNum&)ff[(int) ((f + 1) * 10) ] = 1;
  return ff;
}

/* put me back in a -1 to 1 range */
float label2float(const Feed::Output& layer) {
  typedef typename Net::Num RealNum;
  typedef typename Feed::Num Num;
  int i = max_element(layer.begin(), layer.end(), [](Num a, Num b) { return (RealNum)a < (RealNum)b; }) - layer.begin();
  cout << "Index: " << i << " yielding " << ((i * 2) / (float)num_bins) - 1 << endl;
  return (float)(i * 2) / (float)num_bins - 1;
}

#endif /**** DISCRETE ****/


int main() {
  typedef typename Net::Num RealNum;
  Net net{};
  Feed feed;
  vector< Feed::Layer > X;
  vector< Feed::Output > Y;

  initialize(net, -0.12, 0.12);;

  {
    ifstream file("./noisy-sin.txt");
    RealNum x, y;

    while( file >> x && file >> y) {
      X.push_back( Feed::Layer{{x}} );
      Y.push_back( float2label( y ) );
    }}

  for(size_t i = 0; i < 4000; ++i)
    train(net, X, Y, 0.000001);

  {
    ofstream file("learned-sin.txt");
    for(float x = - 2 * M_PI
	  ; x < (2 * M_PI)
	  ; x += (M_PI / 16)) {
      feed.layer[0] = x;
      // print_array(feed.output_layer()) << endl;

      predict(net,feed);
      label2float( feed.output_layer() );

      file << x << " " << label2float( feed.output_layer() ) << endl;
    }}

  cout << "Done" << endl;

  return 0;
}
