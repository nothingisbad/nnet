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

int main() {
  using namespace std;
  std::random_device rd;
  std::mt19937 gen(rd());

  typedef NNet< array<float,2>
		, array<float,4>
		, array<float,2> >
    Net;

  typedef FoldNet<Net> FoldNet;
  Net net{};
  permute(net, -0.12, 0.12);

  vector< tuple<float,float, int > > D;
  vector< array<float,2> > train_X;
  vector< array<float,2> > train_Y;

  /* read in my data */
  fstream file;
  file.open("data1.txt", fstream::in);
  do {
    D.push_back( make_tuple(0.0f,0.0f, 0) );
  } while( file >> get<0>(D.back()) && file >> get<1>(D.back()) );
  file.close();

  file.open("data2.txt", fstream::in);;
  do {
    D.push_back( make_tuple(0.0f,0.0f, 1) );
  } while( file >> get<0>(D.back()) && file >> get<1>(D.back()) );
  file.close();

  std::uniform_int_distribution<> dist(0, D.size() - 1);

  /* build a training set */
  for(int i = 0; i < 100; ++i) {
    train_X.push_back( array<float,2>{ { get<0>(D[i]), get<1>(D[i]) } } );
    train_Y.push_back( get<2>(D[i]) == 1 ? array<float,2>{{1,-1}} : array<float,2>{{-1,1}}  );
  }

  /* just apply gradient decent a few times and see if the thing worked. */
  float cost;
  Net grad;

  auto error = [](const float& prediction , const float& label ) -> float {
    return (prediction * label) > 0 ? 0 : 1;
  };

  for(int i = 0; i < 100; ++i) {
    cost = cost_function(net, train_X[i], train_Y[i], 3, error);
    grad = gradient(net, train_X[i], train_Y[i], error);

    FoldNet::map( [&cost](float f, float gr) -> float {
	return f + gr * cost;
      }, net, grad);
  }

  fstream file2;

  file.open("predict1.txt", fstream::out);
  file2.open("predict2.txt", fstream::out);

  for_each(D.begin(), D.end()
	   , [&](tuple<float,float,int> d) {
	     auto &f = get<2>(d) > 0 ? file : file2;
	     f << get<0>(d) << " " << get<1>(d) << "\n";
	   });

  return 0;
}
