
/**
 * @file /home/ryan/programming/nnet/make_dist.cpp
 * @author Ryan Domigan <ryan_domigan@sutdents@uml.edu>
 * Created on Apr 08, 2014
 *
 * Make a seperable distribution for use with gnuplot/simple classifier
 */

#include <random>
#include <fstream>
#include <math.h>

using namespace std;

void make_dist(size_t num_samples
	       , mt19937 &gen
	       , uniform_real_distribution<> dist_x
	       , normal_distribution<> perturb_y
	       , ostream &file) {

  for(size_t i = 0; i < num_samples; ++i) {
    float x = dist_x(gen)
      , y = sin(x) + perturb_y(gen);

    file << x << " " << y << "\n";
  }
}

int main() {
  random_device rd;
  mt19937 gen;

  fstream out1, out2;

  out1.open("noisy-sin.txt", std::fstream::out);

  make_dist(301, gen
	    , uniform_real_distribution<>(-2 * M_PI, 2 * M_PI), normal_distribution<>(-0.1,0.1)
	    , out1);
  return 0;
}
