/**
 * @file /home/ryan/programming/nnet/test_load_save.cpp
 * @author Ryan Domigan <ryan_domigan@sutdents@uml.edu>
 * Created on Apr 20, 2014
 *
 * Try storing and retieving node weights from a file
 */

#include "../NNet.hpp"

#include <iostream>
#include <fstream>

int main() {
  using namespace std;
  using namespace recurrence_detail;
  typedef NNet< Nums<1, 2, 1> > Net;

  fstream file;

  /* Specify the networks initial weights */
  Net net{};
  file.open("test.txt", fstream::in);

  read_net(net, file);

  print_network(net);

  return 0;
}

