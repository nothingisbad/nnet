CXX ?= g++
#CXX = clang 
FLAGS := $(FLAGS) -Wall -ggdb -std=c++11 

test_NNet: *.hpp *.cpp
	$(CXX) $(FLAGS) -o test_NNet test_NNet.cpp
