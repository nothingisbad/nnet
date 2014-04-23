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
  Net net( array< array<float, 2>, 2>{{ array<float,2>{{1, 2}}, array<float,2>{{3, 4}} }}

           , array< array<float, 3>, 1>{{ array<float,3>{{1, 2, 3}}  }});

  file.open("test.txt", fstream::out);

  write_net(net, file);

  return 0;
}

