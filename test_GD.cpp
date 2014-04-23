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
#include <iomanip>
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
  std::random_device rd;
  std::mt19937 gen(rd());

  typedef NNet< Nums<2,5,4,2> > Net;
  typedef typename Net::Feed Feed;

  Net net{};
  Feed feed;

  permute(net, -0.12, 0.12);

  print_network(net, cout) << endl;

  data_type D;
  vector< array<float,2> > train_X(150);
  vector< array<float,2> > train_Y(150);

  /* read in my data */
  read_file("data1.txt", D, 0);
  read_file("data2.txt", D, 1);

  std::uniform_int_distribution<> dist(0, D.size() - 1);

  /* build a training set by randomly sampling from D */
  int n;
  for(size_t i = 0; i < train_X.size(); ++i) {
    n = dist(gen);
    train_X[i] = ( array<float,2>{ { get<0>(D[n]), get<1>(D[n]) } } );
    train_Y[i] = ( (get<2>(D[n]) > 0.5 ? array<float,2>{{1,0}} : array<float,2>{{0,1}}) );
  }

  decltype(cost_function(net, train_X, train_Y, 0.2)) cost;
  for(size_t i = 0; i < 1400; ++i) {
    cost = cost_function(net, train_X, train_Y, 0.2);

    if(i % 50 == 0)
      cout << " " << setw(10) << get<1>(cost);

    map([](float &nn, float &grad) {
	nn -= grad;
      }, net, get<0>(cost));
  }

  cout << "\nFinal cost " << get<1>(cost) << "\n";

  fstream file1, file2, file_error
    , file_null;

  file1.open("predict1.txt", fstream::out);
  file2.open("predict2.txt", fstream::out);
  file_error.open("errors.txt", fstream::out);

  file_null.open("/dev/null", fstream::out);

  for(auto dd : D) {
    feed.layer = array<float,2>{{get<0>(dd), get<1>(dd)}};
    predict(net, feed);
    auto p0 = feed.output_layer()[0]
      , p1 = feed.output_layer()[1];

    /* print info for the first and last 10 items */
    // cout << "label: " << get<2>(dd) << " predicted: (" << p0 << ", " << p1 << ")\n";


    auto output = [&](std::ostream &file) {
      file << get<0>(dd) << " " << get<1>(dd) << "\n";
    };

    if(((p0 > 0.5) && (p1 > 0.5)) || ((p0 < 0.5) && (p1 < 0.5)))
      output(file_error);

    else if(feed.output_layer()[0] > 0.5)
      output(file1);

    else
      output(file2);
  }

  return 0;
}
