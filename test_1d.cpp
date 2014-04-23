/**
 * @file /home/ryan/programming/nnet/test_NNet.cpp
 * @author Ryan Domigan <ryan_domigan@sutdents@uml.edu>
 * Created on Apr 06, 2014
 *
 * One dimentional gradient decent test
 */

#include "NNet.hpp"
#include "gradient_decent.hpp"

#include <iostream>
#include <fstream>
#include <tuple>
#include <algorithm>

using namespace std;

typedef vector< tuple<float,float, int > > data_type;

void read_file(const std::string& name, data_type& D, int label) {
  fstream file;
  file.open(name, fstream::in);
  do {
    D.push_back( make_tuple(0.0f,0.0f, label) );
  } while( file >> get<0>(D.back()) && file >> get<1>(D.back()) );
  file.close();
}

int main() {
  using namespace std;
  std::random_device rd;
  std::mt19937 gen(rd());

  typedef NNet< Nums<1,6,1> > Net;
  typedef typename Net::Feed Feed;

  Net net{};
  Feed feed;

  permute(net, -0.12, 0.12);
  print_network(net, cout) << endl;

  data_type D;
  vector< array<float,1> > train_X(100);
  vector< array<float,1> > train_Y(100);
  vector< array<float,1> > predictions(100);

  std::uniform_real_distribution<> dist(0, 1);

  /* build a training set by randomly sampling from D */
  double n;
  for(int i = 0; i < 100; ++i) {
    n = dist(gen);
    train_X[i] = array<float,1>{{(float)n}};
    train_Y[i] = array<float,1>{{ (float)((n > 0.5) ? 1.0 : 0.0) }};
  }


  /* just apply gradient decent a few times and see if the thing worked. */
  auto error = [](float prediction , float label ) -> bool {
    return label > 0.5 ?  (prediction > 0.5) : (prediction < 0.5);
  };


  float old_cost = train_X.size();
  for(int i = 0; i < 100; ++i) {

    auto cost = cost_function(net, train_X, train_Y, 0.5);

    //cout << "* cost: " << get<1>(cost) << endl;;
    if(old_cost <= get<1>(cost)) {
      cout << "COST INCREASE! iteration " << i << endl;

      cout << "\n* Gradient (cost " << get<1>(cost) << ")\n";
      print_network(get<0>(cost), cout) << "\n\n";
      
      cout << "\n* Network \n";
      print_network(net, cout) << "\n\n";
    }
    old_cost = get<1>(cost);

    map([&](float &nn, float &grad) {
    	nn -= grad;
      }, net, get<0>(cost));

  }

  cout << "Final Network: ";
  print_network(net,cout) << "\n\n";

  fstream file1, file2, file_error
    , file_null;

  file1.open("predict1.txt", fstream::out);
  file2.open("predict2.txt", fstream::out);
  file_error.open("errors.txt", fstream::out);

  file_null.open("/dev/null", fstream::out);

  for(size_t i = 0; i < train_X.size(); ++i) {

    feed.layer = train_X[i];
    predict(net, feed);

    /* print info for the first and last 10 items */
    if((i < 10) || (i > (D.size() - 10))) 
      cout << "label: " << train_Y[i][0] << " predicted: " << error(feed.output_layer()[0], train_Y[i][0])
	   << " : " << train_X[i][0] <<  ") -> (" << feed.output_layer()[0] << ")\n";

    auto p0 = feed.output_layer()[0]
      , p1 = feed.output_layer()[1];

    auto output = [&](std::ostream &file) {
      file_null << train_X[i][0] << "\n";
    };

    if(((p0 > 0.5) && (p1 > 0.5)) || ((p0 < 0.5) && (p1 < 0.5))) {
      output(file_error);
    }

    else if(feed.output_layer()[0] > 0.5) {
      output(file1);
    } else {
      output(file2);
    }
  }

  return 0;
}
