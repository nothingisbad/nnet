
/**
 * @file /home/ryan/programming/nnet/make_dist.cpp
 * @author Ryan Domigan <ryan_domigan@sutdents@uml.edu>
 * Created on Apr 08, 2014
 *
 * Make a seperable distribution for use with gnuplot/simple classifier
 */

#include <random>
#include <fstream>

int main() {
  using namespace std;
  random_device rd;
  mt19937 gen;

  fstream out1, out2;

  out1.open("data1.txt", std::fstream::out);
  out2.open("data2.txt", std::fstream::out);

  int samples = 200;

  normal_distribution<> dst_x1(-2,1), dst_y1(2,2)
    , dst_x2(4,1), dst_y2(-4,3);

  for(int i = 0; i < samples; ++i) {
    out1 << dst_x1(gen) << " " << dst_x2(gen) << "\n";
  }

  for(int i = 0; i < samples; ++i) {
    out2 << dst_y1(gen) << " " << dst_x2(gen) << "\n";
  }
}
