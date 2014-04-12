#ifndef ARRAY_HELPERS_HPP
#define ARRAY_HELPERS_HPP
/**
 * @file /home/ryan/programming/nnet/array_helpers.hpp
 * @author Ryan Domigan <ryan_domigan@sutdents@uml.edu>
 * Created on Apr 09, 2014
 */

/* template<size_t i, class Num, class ... Args> */
/* struct MakeArray { */
/*   typedef typename MakeArray<i+1,Args...>::return_type return_type; */
/*   static return_type&& a(Num n, Args ... as) { */
/*     return_type &&tmp = MakeArray<i + 1, Args ...>::a(as ... ); */
/*     tmp[i - 1] = static_cast<float>(n); */
/*     return std::move(tmp); */
/*   } */
/* }; */

/* template<size_t i, class Num> */
/* struct MakeArray<i, Num> { */
/*   typedef std::array<float,i> return_type; */

/*   static std::array<float,i> a(Num n) { */
/*     std::array<float,i> tmp; */
/*     tmp[i - 1] = static_cast<float>(n); */
/*     return tmp; */
/*   } */
/* }; */

/* template<class ... args> */
/* typename MakeArray<1, args ...>::return_type make_array(args ... a) { */
/*   return MakeArray<1, args ...>::a(a ...); */
/* } */




#endif
