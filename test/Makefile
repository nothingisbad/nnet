INCLUDE = -I../../

CXX ?= g++
#CXX = clang -lc++ 
FLAGS := $(FLAGS) -Wall -ggdb -std=c++11 

TARGETS=test_NNet test_xor test_simple test_MetaNet make_sin test_sin make_xor test_1d
$(TARGETS) : % : %.cpp ../*.hpp
	$(CXX) $(FLAGS) $(INCLUDE) -o $@ $@.cpp 

noisy-sin.txt: make_sin
	./make_sin

CLEAN_TARGETS = $(TARGETS) noisy-sin.txt noisy-sine.txt
clean:
	@for item in $(TARGETS); do { if [ -e $${item} ]; then rm $${item} ; echo -n "cleaning up "; echo $${item} ; fi }; done
