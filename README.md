# Usage

Compile time defined, fully interconnected feed forward neural nets.

To define a four layer fully interconnected neural network:

```c++
#include <nnet/NNet.hpp>

typedef NNet<2, 3, 4, 1> Net;
```

# Example

Using a three layer NNet and the included gradient decent training function

```c++
typedef NNet<1,20,20> Net;
for(size_t i = 0; i < 4000; ++i)
    train(net, X, Y, 0.01);
```

I can make a discretized model of a sin wave:

![learned sin wave](https://raw.githubusercontent.com/nothingisbad/nnet/master/test/scaled-learned-sin-wave.png)

see test/test_sin.cpp for the full source, and test/make_sin.cpp for the input generator.
