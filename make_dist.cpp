
/**
 * @file /home/ryan/programming/nnet/make_dist.cpp
 * @author Ryan Domigan <ryan_domigan@sutdents@uml.edu>
 * Created on Apr 08, 2014
 *
 * Make a seperable distribution for use with gnuplot/simple classifier
 */

#include <random>
#include <fstream>

using namespace std;

void make_dist(size_t num_samples
	       , mt19937 &gen
	       , normal_distribution<> dist_x
	       , normal_distribution<> dist_y
	       , ostream &file) {

  for(size_t i = 0; i < num_samples; ++i)
    file << dist_x(gen) << " " << dist_y(gen) << "\n";
}

int main() {
  random_device rd;
  mt19937 gen;

  fstream out1, out2;

  out1.open("data1.txt", std::fstream::out);
  out2.open("data2.txt", std::fstream::out);

  make_dist(100, gen
	    , normal_distribution<>(-1,1), normal_distribution<>(2,8)
	    , out1);

  make_dist(100, gen
	    , normal_distribution<>(1,3), normal_distribution<>(3,2)
	    , out1);

  make_dist(100, gen
  	    , normal_distribution<>(4,1), normal_distribution<>(-4, 3)
  	    , out2);

  make_dist(100, gen
	    , normal_distribution<>(2,1.5), normal_distribution<>(-6, 1)
	    , out2);

}
