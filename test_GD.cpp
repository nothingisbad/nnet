/**
 * @file /home/ryan/programming/nnet/test_NNet.cpp
 * @author Ryan Domigan <ryan_domigan@sutdents@uml.edu>
 * Created on Apr 06, 2014
 *
 * Test gradient decent
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

  typedef NNet< array<float,2>
		, array<float,4>
		, array<float,2> >
    Net;
  typedef typename Net::Feed Feed;

  Net net{};
  Feed feed;

  permute(net, -0.12, 0.12);
  print_network(net, cout) << endl;

  data_type D;
  vector< array<float,2> > train_X(100);
  vector< array<float,2> > train_Y(100);
  vector< array<float,2> > predictions(100);

  /* read in my data */
  read_file("data1.txt", D, 0);
  read_file("data2.txt", D, 1);

  std::uniform_int_distribution<> dist(0, D.size() - 1);

  /* build a training set by randomly sampling from D */
  int n;
  for(int i = 0; i < 100; ++i) {
    n = dist(gen);
    train_X[i] = array<float,2>{ { get<0>(D[n]), get<1>(D[n]) } };
    train_Y[i] = ( get<2>(D[n]) > 0.5 ? array<float,2>{{1,0}} : array<float,2>{{0,1}}  );
  }

  for(int i = 0; i < 10; ++i) {
    cout << "sample " << i << ": ("  << train_Y[i][0] << ", " << train_Y[i][1] << ")\n";
  }

  /* just apply gradient decent a few times and see if the thing worked. */
  auto error = [](float prediction , float label ) -> bool {
    return label < 0.5 ?  (prediction < 0.5) : (prediction > 0.5);
  };

  for(int i = 0; i < 100; ++i) {

    for(int p = 0; p < 100; ++p) {
      feed.layer = train_X[p];
      predict(net, feed);

      predictions[p] = feed.output_layer();

      if((p < 10) && (i == 99))
	cout << "Predicted: (" << train_X[p][0] << ", " << train_X[p][1]
	     << ") -> (" << predictions[p][0] << ", " << predictions[p][1] << ")\n"
	     << "    labled: (" << train_Y[p][0] << ", " << train_Y[p][1] << ")\n";
    }

    auto cost = cost_function(net, train_Y, predictions, error, 0.5);
    //cost = cost_function(net, train_X, train_Y, 1, error);


    if(i < 3) {
      cout << "* Old Gradient (cost " << get<1>(cost) << ")\n";
      print_gradient(net,get<0>(cost)) << "\n";
    }

    apply_gradient(net, get<0>(cost), 0.003);

    if(i < 3) {
      cout << "** New Gradient\n";
      print_gradient(net,get<0>(cost));
    }
  }

  cout << "Gradient: ";
  print_network(net,cout) << endl;

  fstream file1, file2, file_error
    , file_null;

  file1.open("predict1.txt", fstream::out);
  file2.open("predict2.txt", fstream::out);
  file_error.open("errors.txt", fstream::out);

  file_null.open("/dev/null", fstream::out);

  for(size_t i = 0; i < D.size(); ++i) {
    feed.layer = train_X[i];

    predict(net, feed);

    /* print info for the first and last 10 items */
    if(( i++ < 10) || (i > (D.size() - 10))) 
      cout << "label: " << train_Y[i][0] << " predicted: (" << feed.output_layer()[0] << ", " << feed.output_layer()[1] << ")\n";

    auto p0 = feed.output_layer()[0]
      , p1 = feed.output_layer()[1];

    auto output = [&](std::ostream &file) {
      file_null << train_X[i][0] << " " << train_X[i][1] << "\n";
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
