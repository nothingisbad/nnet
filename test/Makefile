CXX ?= g++
#CXX = clang -lc++ 
FLAGS := $(FLAGS) -Wall -ggdb -std=c++11 

test_save: *.cpp ../*.hpp
	$(CXX) $(FLAGS) -o $@ test_save.cpp

test_load: *.cpp ../*.hpp
	$(CXX) $(FLAGS) -o $@ test_load.cpp

TARGETS=test_NNet test_GD test_simple
clean:
	@for item in $(TARGETS); do { if [ -e $${item} ]; then rm $${item} ; echo -n "cleaning up "; echo $${item} ; fi }; done
