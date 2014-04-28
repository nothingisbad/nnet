#ifndef FIXEDPOINT_HPP
#define FIXEDPOINT_HPP
/**
 * @file /home/ryan/programming/nnet/FixedPoint.hpp
 * Ryan Domigan <ryan_domigan@sutdents@uml.edu>
 * 
 * Modifying to follow my own wacky coding standards.
 * Updated on Apr 22, 2014
 *
 *  Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 * (C) Copyright 2007 Anthony Williams
 */

#include "fixed.hpp"
#ifndef FIXED_HPP
#define FIXED_HPP

#include <ostream>
#include <complex>

unsigned const fixed_resolution_shift=28;
__int64 const fixed_resolution = 1I64 << fixed_resolution_shift;

template<class Input, class Base, size_t E>
struct Convert
  : std::conditional< std::is_integral<Base>::value
		      , ConvertInt<Input, Base, E>
		      , ConvertFloat<Input, Base, E>
		      >::type {};

templat <class Base, size_t Exponent>
class fixed {
private:
  Base _m_nVal;
  const static Base _fractional_mask = -1 << fixed_resolution_shift;
  
public:
  struct internal {};

  fixed(): m_nVal(0) {}
    
  fixed(internal, Base nVal) :  m_nVal(nVal) {}

  template<class Num>    
  fixed(Num nVal) : m_nVal(Base(nVal) << Exponent)  {}
    
  fixed(double nVal):
    m_nVal(static_cast<__int64>(nVal*static_cast<double>(fixed_resolution)))
  {}
  fixed(float nVal):
    m_nVal(static_cast<__int64>(nVal*static_cast<float>(fixed_resolution)))
  {}

  template<typename T>
  fixed& operator=(T other)
  {
    m_nVal=fixed(other).m_nVal;
    return *this;
  }
  fixed& operator=(fixed const& other)
  {
    m_nVal=other.m_nVal;
    return *this;
  }
  friend bool operator==(fixed const& lhs,fixed const& rhs)
  {
    return lhs.m_nVal==rhs.m_nVal;
  }
  friend bool operator!=(fixed const& lhs,fixed const& rhs)
  {
    return lhs.m_nVal!=rhs.m_nVal;
  }
  friend bool operator<(fixed const& lhs,fixed const& rhs)
  {
    return lhs.m_nVal<rhs.m_nVal;
  }
  friend bool operator>(fixed const& lhs,fixed const& rhs)
  {
    return lhs.m_nVal>rhs.m_nVal;
  }
  friend bool operator<=(fixed const& lhs,fixed const& rhs)
  {
    return lhs.m_nVal<=rhs.m_nVal;
  }
  friend bool operator>=(fixed const& lhs,fixed const& rhs)
  {
    return lhs.m_nVal>=rhs.m_nVal;
  }
  operator bool() const
  {
    return m_nVal?true:false;
  }
  inline operator double() const
  {
    return as_double();
  }
  float as_float() const
  {
    return m_nVal/(float)fixed_resolution;
  }

  double as_double() const
  {
    return m_nVal/(double)fixed_resolution;
  }

  long as_long() const
  {
    return (long)(m_nVal/fixed_resolution);
  }
  __int64 as_int64() const
  {
    return m_nVal/fixed_resolution;
  }

  int as_int() const
  {
    return (int)(m_nVal/fixed_resolution);
  }

  unsigned long as_unsigned_long() const
  {
    return (unsigned long)(m_nVal/fixed_resolution);
  }
  unsigned __int64 as_unsigned_int64() const
  {
    return (unsigned __int64)m_nVal/fixed_resolution;
  }

  unsigned int as_unsigned_int() const
  {
    return (unsigned int)(m_nVal/fixed_resolution);
  }

  short as_short() const
  {
    return (short)(m_nVal/fixed_resolution);
  }

  unsigned short as_unsigned_short() const
  {
    return (unsigned short)(m_nVal/fixed_resolution);
  }

  fixed operator++()
  {
    m_nVal += fixed_resolution;
    return *this;
  }

  fixed operator--()
  {
    m_nVal -= fixed_resolution;
    return *this;
  }

  fixed floor() const;
  fixed ceil() const;
  fixed sqrt() const;
  fixed exp() const;
  fixed log() const;
  fixed& operator%=(fixed const& other);
  fixed& operator*=(fixed const& val);
  fixed& operator/=(fixed const& val);
  fixed& operator-=(fixed const& val)
  {
    m_nVal -= val.m_nVal;
    return *this;
  }

  fixed& operator+=(fixed const& val)
  {
    m_nVal += val.m_nVal;
    return *this;
  }
  fixed& operator*=(double val)
  {
    return (*this)*=fixed(val);
  }
  fixed& operator*=(float val)
  {
    return (*this)*=fixed(val);
  }
  fixed& operator*=(__int64 val)
  {
    m_nVal*=val;
    return *this;
  }
  fixed& operator*=(long val)
  {
    m_nVal*=val;
    return *this;
  }
  fixed& operator*=(int val)
  {
    m_nVal*=val;
    return *this;
  }
  fixed& operator*=(short val)
  {
    m_nVal*=val;
    return *this;
  }
  fixed& operator*=(char val)
  {
    m_nVal*=val;
    return *this;
  }
  fixed& operator*=(unsigned __int64 val)
  {
    m_nVal*=val;
    return *this;
  }
  fixed& operator*=(unsigned long val)
  {
    m_nVal*=val;
    return *this;
  }
  fixed& operator*=(unsigned int val)
  {
    m_nVal*=val;
    return *this;
  }
  fixed& operator*=(unsigned short val)
  {
    m_nVal*=val;
    return *this;
  }
  fixed& operator*=(unsigned char val)
  {
    m_nVal*=val;
    return *this;
  }
  fixed& operator/=(double val)
  {
    return (*this)/=fixed(val);
  }
  fixed& operator/=(float val)
  {
    return (*this)/=fixed(val);
  }
  fixed& operator/=(__int64 val)
  {
    m_nVal/=val;
    return *this;
  }
  fixed& operator/=(long val)
  {
    m_nVal/=val;
    return *this;
  }
  fixed& operator/=(int val)
  {
    m_nVal/=val;
    return *this;
  }
  fixed& operator/=(short val)
  {
    m_nVal/=val;
    return *this;
  }
  fixed& operator/=(char val)
  {
    m_nVal/=val;
    return *this;
  }
  fixed& operator/=(unsigned __int64 val)
  {
    m_nVal/=val;
    return *this;
  }
  fixed& operator/=(unsigned long val)
  {
    m_nVal/=val;
    return *this;
  }
  fixed& operator/=(unsigned int val)
  {
    m_nVal/=val;
    return *this;
  }
  fixed& operator/=(unsigned short val)
  {
    m_nVal/=val;
    return *this;
  }
  fixed& operator/=(unsigned char val)
  {
    m_nVal/=val;
    return *this;
  }
    

  bool operator!() const
  {
    return m_nVal==0;
  }
    
  fixed modf(fixed* integral_part) const;
  fixed atan() const;

  static void sin_cos(fixed const& theta,fixed* s,fixed*c);
  static void to_polar(fixed const& x,fixed const& y,fixed* r,fixed*theta);

  fixed sin() const;
  fixed cos() const;
  fixed tan() const;
  fixed operator-() const;
  fixed abs() const;
};

inline std::ostream& operator<<(std::ostream& os,fixed const& value) {
  return os<<value.as_double();
}

inline fixed operator-(double a, fixed const& b) {
  fixed temp(a);
  return temp-=b;
}


inline fixed operator-(float a, fixed const& b) {
  fixed temp(a);
  return temp-=b;
}

inline fixed operator-(unsigned long a, fixed const& b) {
  fixed temp(a);
  return temp-=b;
}

inline fixed operator-(long a, fixed const& b) {
  fixed temp(a);
  return temp-=b;
}

inline fixed operator-(unsigned a, fixed const& b) {
  fixed temp(a);
  return temp-=b;
}

inline fixed operator-(int a, fixed const& b) {
  fixed temp(a);
  return temp-=b;
}

inline fixed operator-(unsigned short a, fixed const& b) {
  fixed temp(a);
  return temp-=b;
}

inline fixed operator-(short a, fixed const& b) {
  fixed temp(a);
  return temp-=b;
}

inline fixed operator-(unsigned char a, fixed const& b) {
  fixed temp(a);
  return temp-=b;
}

inline fixed operator-(char a, fixed const& b) {
  fixed temp(a);
  return temp-=b;
}

inline fixed operator-(fixed const& a,double b) {
  fixed temp(a);
  return temp-=b;
}


inline fixed operator-(fixed const& a,float b) {
  fixed temp(a);
  return temp-=b;
}

inline fixed operator-(fixed const& a,unsigned long b) {
  fixed temp(a);
  return temp-=b;
}

inline fixed operator-(fixed const& a,long b) {
  fixed temp(a);
  return temp-=b;
}

inline fixed operator-(fixed const& a,unsigned b) {
  fixed temp(a);
  return temp-=b;
}

inline fixed operator-(fixed const& a,int b) {
  fixed temp(a);
  return temp-=b;
}

inline fixed operator-(fixed const& a,unsigned short b) {
  fixed temp(a);
  return temp-=b;
}

inline fixed operator-(fixed const& a,short b) {
  fixed temp(a);
  return temp-=b;
}

inline fixed operator-(fixed const& a,unsigned char b) {
  fixed temp(a);
  return temp-=b;
}

inline fixed operator-(fixed const& a,char b) {
  fixed temp(a);
  return temp-=b;
}

inline fixed operator-(fixed const& a,fixed const& b) {
  fixed temp(a);
  return temp-=b;
}

inline fixed operator%(double a, fixed const& b) {
  fixed temp(a);
  return temp%=b;
}


inline fixed operator%(float a, fixed const& b) {
  fixed temp(a);
  return temp%=b;
}

inline fixed operator%(unsigned long a, fixed const& b) {
  fixed temp(a);
  return temp%=b;
}

inline fixed operator%(long a, fixed const& b) {
  fixed temp(a);
  return temp%=b;
}

inline fixed operator%(unsigned a, fixed const& b) {
  fixed temp(a);
  return temp%=b;
}

inline fixed operator%(int a, fixed const& b) {
  fixed temp(a);
  return temp%=b;
}

inline fixed operator%(unsigned short a, fixed const& b) {
  fixed temp(a);
  return temp%=b;
}

inline fixed operator%(short a, fixed const& b) {
  fixed temp(a);
  return temp%=b;
}

inline fixed operator%(unsigned char a, fixed const& b) {
  fixed temp(a);
  return temp%=b;
}

inline fixed operator%(char a, fixed const& b) {
  fixed temp(a);
  return temp%=b;
}

inline fixed operator%(fixed const& a,double b) {
  fixed temp(a);
  return temp%=b;
}


inline fixed operator%(fixed const& a,float b) {
  fixed temp(a);
  return temp%=b;
}

inline fixed operator%(fixed const& a,unsigned long b) {
  fixed temp(a);
  return temp%=b;
}

inline fixed operator%(fixed const& a,long b) {
  fixed temp(a);
  return temp%=b;
}

inline fixed operator%(fixed const& a,unsigned b) {
  fixed temp(a);
  return temp%=b;
}

inline fixed operator%(fixed const& a,int b) {
  fixed temp(a);
  return temp%=b;
}

inline fixed operator%(fixed const& a,unsigned short b) {
  fixed temp(a);
  return temp%=b;
}

inline fixed operator%(fixed const& a,short b) {
  fixed temp(a);
  return temp%=b;
}

inline fixed operator%(fixed const& a,unsigned char b) {
  fixed temp(a);
  return temp%=b;
}

inline fixed operator%(fixed const& a,char b) {
  fixed temp(a);
  return temp%=b;
}

inline fixed operator%(fixed const& a,fixed const& b) {
  fixed temp(a);
  return temp%=b;
}

inline fixed operator+(double a, fixed const& b) {
  fixed temp(a);
  return temp+=b;
}


inline fixed operator+(float a, fixed const& b) {
  fixed temp(a);
  return temp+=b;
}

inline fixed operator+(unsigned long a, fixed const& b) {
  fixed temp(a);
  return temp+=b;
}

inline fixed operator+(long a, fixed const& b) {
  fixed temp(a);
  return temp+=b;
}

inline fixed operator+(unsigned a, fixed const& b) {
  fixed temp(a);
  return temp+=b;
}

inline fixed operator+(int a, fixed const& b) {
  fixed temp(a);
  return temp+=b;
}

inline fixed operator+(unsigned short a, fixed const& b) {
  fixed temp(a);
  return temp+=b;
}

inline fixed operator+(short a, fixed const& b) {
  fixed temp(a);
  return temp+=b;
}

inline fixed operator+(unsigned char a, fixed const& b) {
  fixed temp(a);
  return temp+=b;
}

inline fixed operator+(char a, fixed const& b) {
  fixed temp(a);
  return temp+=b;
}

inline fixed operator+(fixed const& a,double b) {
  fixed temp(a);
  return temp+=b;
}


inline fixed operator+(fixed const& a,float b) {
  fixed temp(a);
  return temp+=b;
}

inline fixed operator+(fixed const& a,unsigned long b) {
  fixed temp(a);
  return temp+=b;
}

inline fixed operator+(fixed const& a,long b) {
  fixed temp(a);
  return temp+=b;
}

inline fixed operator+(fixed const& a,unsigned b) {
  fixed temp(a);
  return temp+=b;
}

inline fixed operator+(fixed const& a,int b) {
  fixed temp(a);
  return temp+=b;
}

inline fixed operator+(fixed const& a,unsigned short b) {
  fixed temp(a);
  return temp+=b;
}

inline fixed operator+(fixed const& a,short b) {
  fixed temp(a);
  return temp+=b;
}

inline fixed operator+(fixed const& a,unsigned char b) {
  fixed temp(a);
  return temp+=b;
}

inline fixed operator+(fixed const& a,char b) {
  fixed temp(a);
  return temp+=b;
}

inline fixed operator+(fixed const& a,fixed const& b) {
  fixed temp(a);
  return temp+=b;
}

inline fixed operator*(double a, fixed const& b) {
  fixed temp(b);
  return temp*=a;
}


inline fixed operator*(float a, fixed const& b) {
  fixed temp(b);
  return temp*=a;
}

inline fixed operator*(unsigned long a, fixed const& b) {
  fixed temp(b);
  return temp*=a;
}

inline fixed operator*(long a, fixed const& b) {
  fixed temp(b);
  return temp*=a;
}

inline fixed operator*(unsigned a, fixed const& b) {
  fixed temp(b);
  return temp*=a;
}

inline fixed operator*(int a, fixed const& b) {
  fixed temp(b);
  return temp*=a;
}

inline fixed operator*(unsigned short a, fixed const& b) {
  fixed temp(b);
  return temp*=a;
}

inline fixed operator*(short a, fixed const& b) {
  fixed temp(b);
  return temp*=a;
}

inline fixed operator*(unsigned char a, fixed const& b) {
  fixed temp(b);
  return temp*=a;
}

inline fixed operator*(char a, fixed const& b) {
  fixed temp(b);
  return temp*=a;
}

inline fixed operator*(fixed const& a,double b) {
  fixed temp(a);
  return temp*=b;
}


inline fixed operator*(fixed const& a,float b) {
  fixed temp(a);
  return temp*=b;
}

inline fixed operator*(fixed const& a,unsigned long b) {
  fixed temp(a);
  return temp*=b;
}

inline fixed operator*(fixed const& a,long b) {
  fixed temp(a);
  return temp*=b;
}

inline fixed operator*(fixed const& a,unsigned b) {
  fixed temp(a);
  return temp*=b;
}

inline fixed operator*(fixed const& a,int b) {
  fixed temp(a);
  return temp*=b;
}

inline fixed operator*(fixed const& a,unsigned short b) {
  fixed temp(a);
  return temp*=b;
}

inline fixed operator*(fixed const& a,short b) {
  fixed temp(a);
  return temp*=b;
}

inline fixed operator*(fixed const& a,unsigned char b) {
  fixed temp(a);
  return temp*=b;
}

inline fixed operator*(fixed const& a,char b) {
  fixed temp(a);
  return temp*=b;
}

inline fixed operator*(fixed const& a,fixed const& b) {
  fixed temp(a);
  return temp*=b;
}

inline fixed operator/(double a, fixed const& b) {
  fixed temp(a);
  return temp/=b;
}


inline fixed operator/(float a, fixed const& b) {
  fixed temp(a);
  return temp/=b;
}

inline fixed operator/(unsigned long a, fixed const& b) {
  fixed temp(a);
  return temp/=b;
}

inline fixed operator/(long a, fixed const& b) {
  fixed temp(a);
  return temp/=b;
}

inline fixed operator/(unsigned a, fixed const& b) {
  fixed temp(a);
  return temp/=b;
}

inline fixed operator/(int a, fixed const& b) {
  fixed temp(a);
  return temp/=b;
}

inline fixed operator/(unsigned short a, fixed const& b) {
  fixed temp(a);
  return temp/=b;
}

inline fixed operator/(short a, fixed const& b) {
  fixed temp(a);
  return temp/=b;
}

inline fixed operator/(unsigned char a, fixed const& b) {
  fixed temp(a);
  return temp/=b;
}

inline fixed operator/(char a, fixed const& b) {
  fixed temp(a);
  return temp/=b;
}

inline fixed operator/(fixed const& a,double b) {
  fixed temp(a);
  return temp/=b;
}


inline fixed operator/(fixed const& a,float b) {
  fixed temp(a);
  return temp/=b;
}

inline fixed operator/(fixed const& a,unsigned long b) {
  fixed temp(a);
  return temp/=b;
}

inline fixed operator/(fixed const& a,long b) {
  fixed temp(a);
  return temp/=b;
}

inline fixed operator/(fixed const& a,unsigned b) {
  fixed temp(a);
  return temp/=b;
}

inline fixed operator/(fixed const& a,int b) {
  fixed temp(a);
  return temp/=b;
}

inline fixed operator/(fixed const& a,unsigned short b) {
  fixed temp(a);
  return temp/=b;
}

inline fixed operator/(fixed const& a,short b) {
  fixed temp(a);
  return temp/=b;
}

inline fixed operator/(fixed const& a,unsigned char b) {
  fixed temp(a);
  return temp/=b;
}

inline fixed operator/(fixed const& a,char b) {
  fixed temp(a);
  return temp/=b;
}

inline fixed operator/(fixed const& a,fixed const& b) {
  fixed temp(a);
  return temp/=b;
}

inline bool operator==(double a, fixed const& b) {
  return fixed(a)==b;
}
inline bool operator==(float a, fixed const& b) {
  return fixed(a)==b;
}

inline bool operator==(unsigned long a, fixed const& b) {
  return fixed(a)==b;
}
inline bool operator==(long a, fixed const& b) {
  return fixed(a)==b;
}
inline bool operator==(unsigned a, fixed const& b) {
  return fixed(a)==b;
}
inline bool operator==(int a, fixed const& b) {
  return fixed(a)==b;
}
inline bool operator==(unsigned short a, fixed const& b) {
  return fixed(a)==b;
}
inline bool operator==(short a, fixed const& b) {
  return fixed(a)==b;
}
inline bool operator==(unsigned char a, fixed const& b) {
  return fixed(a)==b;
}
inline bool operator==(char a, fixed const& b) {
  return fixed(a)==b;
}

inline bool operator==(fixed const& a,double b) {
  return a==fixed(b);
}
inline bool operator==(fixed const& a,float b) {
  return a==fixed(b);
}
inline bool operator==(fixed const& a,unsigned long b) {
  return a==fixed(b);
}
inline bool operator==(fixed const& a,long b) {
  return a==fixed(b);
}
inline bool operator==(fixed const& a,unsigned b) {
  return a==fixed(b);
}
inline bool operator==(fixed const& a,int b) {
  return a==fixed(b);
}
inline bool operator==(fixed const& a,unsigned short b) {
  return a==fixed(b);
}
inline bool operator==(fixed const& a,short b) {
  return a==fixed(b);
}
inline bool operator==(fixed const& a,unsigned char b) {
  return a==fixed(b);
}
inline bool operator==(fixed const& a,char b) {
  return a==fixed(b);
}

inline bool operator!=(double a, fixed const& b) {
  return fixed(a)!=b;
}
inline bool operator!=(float a, fixed const& b) {
  return fixed(a)!=b;
}

inline bool operator!=(unsigned long a, fixed const& b) {
  return fixed(a)!=b;
}
inline bool operator!=(long a, fixed const& b) {
  return fixed(a)!=b;
}
inline bool operator!=(unsigned a, fixed const& b) {
  return fixed(a)!=b;
}
inline bool operator!=(int a, fixed const& b) {
  return fixed(a)!=b;
}
inline bool operator!=(unsigned short a, fixed const& b) {
  return fixed(a)!=b;
}
inline bool operator!=(short a, fixed const& b) {
  return fixed(a)!=b;
}
inline bool operator!=(unsigned char a, fixed const& b) {
  return fixed(a)!=b;
}
inline bool operator!=(char a, fixed const& b) {
  return fixed(a)!=b;
}

inline bool operator!=(fixed const& a,double b) {
  return a!=fixed(b);
}
inline bool operator!=(fixed const& a,float b) {
  return a!=fixed(b);
}
inline bool operator!=(fixed const& a,unsigned long b) {
  return a!=fixed(b);
}
inline bool operator!=(fixed const& a,long b) {
  return a!=fixed(b);
}
inline bool operator!=(fixed const& a,unsigned b) {
  return a!=fixed(b);
}
inline bool operator!=(fixed const& a,int b) {
  return a!=fixed(b);
}
inline bool operator!=(fixed const& a,unsigned short b) {
  return a!=fixed(b);
}
inline bool operator!=(fixed const& a,short b) {
  return a!=fixed(b);
}
inline bool operator!=(fixed const& a,unsigned char b) {
  return a!=fixed(b);
}
inline bool operator!=(fixed const& a,char b) {
  return a!=fixed(b);
}

inline bool operator<(double a, fixed const& b) {
  return fixed(a)<b;
}
inline bool operator<(float a, fixed const& b) {
  return fixed(a)<b;
}

inline bool operator<(unsigned long a, fixed const& b) {
  return fixed(a)<b;
}
inline bool operator<(long a, fixed const& b) {
  return fixed(a)<b;
}
inline bool operator<(unsigned a, fixed const& b) {
  return fixed(a)<b;
}
inline bool operator<(int a, fixed const& b) {
  return fixed(a)<b;
}
inline bool operator<(unsigned short a, fixed const& b) {
  return fixed(a)<b;
}
inline bool operator<(short a, fixed const& b) {
  return fixed(a)<b;
}
inline bool operator<(unsigned char a, fixed const& b) {
  return fixed(a)<b;
}
inline bool operator<(char a, fixed const& b) {
  return fixed(a)<b;
}

inline bool operator<(fixed const& a,double b) {
  return a<fixed(b);
}
inline bool operator<(fixed const& a,float b) {
  return a<fixed(b);
}
inline bool operator<(fixed const& a,unsigned long b) {
  return a<fixed(b);
}
inline bool operator<(fixed const& a,long b) {
  return a<fixed(b);
}
inline bool operator<(fixed const& a,unsigned b) {
  return a<fixed(b);
}
inline bool operator<(fixed const& a,int b) {
  return a<fixed(b);
}
inline bool operator<(fixed const& a,unsigned short b) {
  return a<fixed(b);
}
inline bool operator<(fixed const& a,short b) {
  return a<fixed(b);
}
inline bool operator<(fixed const& a,unsigned char b) {
  return a<fixed(b);
}
inline bool operator<(fixed const& a,char b) {
  return a<fixed(b);
}

inline bool operator>(double a, fixed const& b) {
  return fixed(a)>b;
}
inline bool operator>(float a, fixed const& b) {
  return fixed(a)>b;
}

inline bool operator>(unsigned long a, fixed const& b) {
  return fixed(a)>b;
}
inline bool operator>(long a, fixed const& b) {
  return fixed(a)>b;
}
inline bool operator>(unsigned a, fixed const& b) {
  return fixed(a)>b;
}
inline bool operator>(int a, fixed const& b) {
  return fixed(a)>b;
}
inline bool operator>(unsigned short a, fixed const& b) {
  return fixed(a)>b;
}
inline bool operator>(short a, fixed const& b) {
  return fixed(a)>b;
}
inline bool operator>(unsigned char a, fixed const& b) {
  return fixed(a)>b;
}
inline bool operator>(char a, fixed const& b) {
  return fixed(a)>b;
}

inline bool operator>(fixed const& a,double b) {
  return a>fixed(b);
}
inline bool operator>(fixed const& a,float b) {
  return a>fixed(b);
}
inline bool operator>(fixed const& a,unsigned long b) {
  return a>fixed(b);
}
inline bool operator>(fixed const& a,long b) {
  return a>fixed(b);
}
inline bool operator>(fixed const& a,unsigned b) {
  return a>fixed(b);
}
inline bool operator>(fixed const& a,int b) {
  return a>fixed(b);
}
inline bool operator>(fixed const& a,unsigned short b) {
  return a>fixed(b);
}
inline bool operator>(fixed const& a,short b) {
  return a>fixed(b);
}
inline bool operator>(fixed const& a,unsigned char b) {
  return a>fixed(b);
}
inline bool operator>(fixed const& a,char b) {
  return a>fixed(b);
}

inline bool operator<=(double a, fixed const& b) {
  return fixed(a)<=b;
}
inline bool operator<=(float a, fixed const& b) {
  return fixed(a)<=b;
}

inline bool operator<=(unsigned long a, fixed const& b) {
  return fixed(a)<=b;
}
inline bool operator<=(long a, fixed const& b) {
  return fixed(a)<=b;
}
inline bool operator<=(unsigned a, fixed const& b) {
  return fixed(a)<=b;
}
inline bool operator<=(int a, fixed const& b) {
  return fixed(a)<=b;
}
inline bool operator<=(unsigned short a, fixed const& b) {
  return fixed(a)<=b;
}
inline bool operator<=(short a, fixed const& b) {
  return fixed(a)<=b;
}
inline bool operator<=(unsigned char a, fixed const& b) {
  return fixed(a)<=b;
}
inline bool operator<=(char a, fixed const& b) {
  return fixed(a)<=b;
}

inline bool operator<=(fixed const& a,double b) {
  return a<=fixed(b);
}
inline bool operator<=(fixed const& a,float b) {
  return a<=fixed(b);
}
inline bool operator<=(fixed const& a,unsigned long b) {
  return a<=fixed(b);
}
inline bool operator<=(fixed const& a,long b) {
  return a<=fixed(b);
}
inline bool operator<=(fixed const& a,unsigned b) {
  return a<=fixed(b);
}
inline bool operator<=(fixed const& a,int b) {
  return a<=fixed(b);
}
inline bool operator<=(fixed const& a,unsigned short b) {
  return a<=fixed(b);
}
inline bool operator<=(fixed const& a,short b) {
  return a<=fixed(b);
}
inline bool operator<=(fixed const& a,unsigned char b) {
  return a<=fixed(b);
}
inline bool operator<=(fixed const& a,char b) {
  return a<=fixed(b);
}

inline bool operator>=(double a, fixed const& b) {
  return fixed(a)>=b;
}
inline bool operator>=(float a, fixed const& b) {
  return fixed(a)>=b;
}

inline bool operator>=(unsigned long a, fixed const& b) {
  return fixed(a)>=b;
}
inline bool operator>=(long a, fixed const& b) {
  return fixed(a)>=b;
}
inline bool operator>=(unsigned a, fixed const& b) {
  return fixed(a)>=b;
}
inline bool operator>=(int a, fixed const& b) {
  return fixed(a)>=b;
}
inline bool operator>=(unsigned short a, fixed const& b) {
  return fixed(a)>=b;
}
inline bool operator>=(short a, fixed const& b) {
  return fixed(a)>=b;
}
inline bool operator>=(unsigned char a, fixed const& b) {
  return fixed(a)>=b;
}
inline bool operator>=(char a, fixed const& b) {
  return fixed(a)>=b;
}

inline bool operator>=(fixed const& a,double b) {
  return a>=fixed(b);
}
inline bool operator>=(fixed const& a,float b) {
  return a>=fixed(b);
}
inline bool operator>=(fixed const& a,unsigned long b) {
  return a>=fixed(b);
}
inline bool operator>=(fixed const& a,long b) {
  return a>=fixed(b);
}
inline bool operator>=(fixed const& a,unsigned b) {
  return a>=fixed(b);
}
inline bool operator>=(fixed const& a,int b) {
  return a>=fixed(b);
}
inline bool operator>=(fixed const& a,unsigned short b) {
  return a>=fixed(b);
}
inline bool operator>=(fixed const& a,short b) {
  return a>=fixed(b);
}
inline bool operator>=(fixed const& a,unsigned char b) {
  return a>=fixed(b);
}
inline bool operator>=(fixed const& a,char b) {
  return a>=fixed(b);
}

inline fixed sin(fixed const& x) {
  return x.sin();
}
inline fixed cos(fixed const& x) {
  return x.cos();
}
inline fixed tan(fixed const& x) {
  return x.tan();
}

inline fixed sqrt(fixed const& x) {
  return x.sqrt();
}

inline fixed exp(fixed const& x) {
  return x.exp();
}

inline fixed log(fixed const& x) {
  return x.log();
}

inline fixed floor(fixed const& x) {
  return x.floor();
}

inline fixed ceil(fixed const& x) {
  return x.ceil();
}

inline fixed abs(fixed const& x) {
  return x.abs();
}

inline fixed modf(fixed const& x,fixed*integral_part) {
  return x.modf(integral_part);
}

inline fixed fixed::ceil() const {
  if(m_nVal%fixed_resolution)
    {
      return floor()+1;
    }
  else
    {
      return *this;
    }
}

inline fixed fixed::floor() const {
  fixed res(*this);
  __int64 const remainder=m_nVal%fixed_resolution;
  if(remainder)
    {
      res.m_nVal-=remainder;
      if(m_nVal<0)
        {
	  res-=1;
        }
    }
  return res;
}


inline fixed fixed::sin() const {
  fixed res;
  sin_cos(*this,&res,0);
  return res;
}

inline fixed fixed::cos() const {
  fixed res;
  sin_cos(*this,0,&res);
  return res;
}

inline fixed fixed::tan() const {
  fixed s,c;
  sin_cos(*this,&s,&c);
  return s/c;
}

inline fixed fixed::operator-() const {
  return fixed(internal(),-m_nVal);
}

inline fixed fixed::abs() const {
  return fixed(internal(),m_nVal<0?-m_nVal:m_nVal);
}

inline fixed fixed::modf(fixed*integral_part) const {
  __int64 fractional_part=m_nVal%fixed_resolution;
  if(m_nVal<0 && fractional_part>0)
    {
      fractional_part-=fixed_resolution;
    }
  integral_part->m_nVal=m_nVal-fractional_part;
  return fixed(internal(),fractional_part);
}

namespace std {
  template<>
  inline ::fixed arg(const std::complex<::fixed>& val)
  {
    ::fixed r,theta;
    ::fixed::to_polar(val.real(),val.imag(),&r,&theta);
    return theta;
  }

  template<>
  inline complex<::fixed> polar(::fixed const& rho,::fixed const& theta) {
    ::fixed s,c;
    ::fixed::sin_cos(theta,&s,&c);
    return complex<::fixed>(rho * c, rho * s);
  }
}

fixed const fixed_max(fixed::internal(),0x7fffffffffffffffI64);
fixed const fixed_one(fixed::internal(),1I64<<(fixed_resolution_shift));
fixed const fixed_zero(fixed::internal(),0);
fixed const fixed_half(fixed::internal(),1I64<<(fixed_resolution_shift-1));
extern fixed const fixed_pi;
extern fixed const fixed_two_pi;
extern fixed const fixed_half_pi;
extern fixed const fixed_quarter_pi;

#endif

__int64 const internal_pi=0x3243f6a8;
__int64 const internal_two_pi=0x6487ed51;
__int64 const internal_half_pi=0x1921fb54;
__int64 const internal_quarter_pi=0xc90fdaa;

extern fixed const fixed_pi(fixed::internal(),internal_pi);
extern fixed const fixed_two_pi(fixed::internal(),internal_two_pi);
extern fixed const fixed_half_pi(fixed::internal(),internal_half_pi);
extern fixed const fixed_quarter_pi(fixed::internal(),internal_quarter_pi);

fixed& fixed::operator%=(fixed const& other) {
  m_nVal = m_nVal%other.m_nVal;
  return *this;
}

fixed& fixed::operator*=(fixed const& val) {
  bool const val_negative=val.m_nVal<0;
  bool const this_negative=m_nVal<0;
  bool const negate=val_negative ^ this_negative;
  unsigned __int64 const other=val_negative?-val.m_nVal:val.m_nVal;
  unsigned __int64 const self=this_negative?-m_nVal:m_nVal;
    
  if(unsigned __int64 const self_upper=(self>>32)) {
    m_nVal=(self_upper*other)<<(32-fixed_resolution_shift);
    else
      m_nVal=0;

    if(unsigned __int64 const self_lower=(self&0xffffffff)) {
      unsigned long const other_upper=static_cast<unsigned long>(other>>32);
      unsigned long const other_lower=static_cast<unsigned long>(other&0xffffffff);
      unsigned __int64 const lower_self_upper_other_res=self_lower*other_upper;
      unsigned __int64 const lower_self_lower_other_res=self_lower*other_lower;
      m_nVal+=(lower_self_upper_other_res<<(32-fixed_resolution_shift))
	+ (lower_self_lower_other_res>>fixed_resolution_shift);
    }
    
    if(negate)
      m_nVal=-m_nVal;

    return *this;
  }


  fixed& fixed::operator/=(fixed const& divisor) {
    if( !divisor.m_nVal)
      m_nVal=fixed_max.m_nVal;

    else {
      bool const negate_this=(m_nVal<0);
      bool const negate_divisor=(divisor.m_nVal<0);
      bool const negate=negate_this ^ negate_divisor;
      unsigned __int64 a=negate_this?-m_nVal:m_nVal;
      unsigned __int64 b=negate_divisor?-divisor.m_nVal:divisor.m_nVal;

      unsigned __int64 res=0;
    
      unsigned __int64 temp=b;
      bool const a_large=a>b;
      unsigned shift=fixed_resolution_shift;

      if(a_large)
	{
	  unsigned __int64 const half_a=a>>1;
	  while(temp<half_a)
	    {
	      temp<<=1;
	      ++shift;
	    }
	}
      unsigned __int64 d=1I64<<shift;
      if(a_large)
	{
	  a-=temp;
	  res+=d;
	}

      while(a && temp && shift)
	{
	  unsigned right_shift=0;
	  while(right_shift<shift && (temp>a))
	    {
	      temp>>=1;
	      ++right_shift;
	    }
	  d>>=right_shift;
	  shift-=right_shift;
	  a-=temp;
	  res+=d;
	}
      m_nVal=(negate?-(__int64)res:res);
    }
    
    return *this;
  }


  fixed fixed::sqrt() const {
    unsigned const max_shift=62;
    unsigned __int64 a_squared=1I64<<max_shift;
    unsigned b_shift=(max_shift+fixed_resolution_shift)/2;
    unsigned __int64 a=1I64<<b_shift;
    
    unsigned __int64 x=m_nVal;
    
    while(b_shift && a_squared>x)
      {
        a>>=1;
        a_squared>>=2;
        --b_shift;
      }

    unsigned __int64 remainder=x-a_squared;
    --b_shift;
    
    while(remainder && b_shift)
      {
        unsigned __int64 b_squared=1I64<<(2*b_shift-fixed_resolution_shift);
        int const two_a_b_shift=b_shift+1-fixed_resolution_shift;
        unsigned __int64 two_a_b=(two_a_b_shift>0)?(a<<two_a_b_shift):(a>>-two_a_b_shift);
        
        while(b_shift && remainder<(b_squared+two_a_b))
	  {
            b_squared>>=2;
            two_a_b>>=1;
            --b_shift;
	  }
        unsigned __int64 const delta=b_squared+two_a_b;
        if((2*remainder)>delta)
	  {
            a+=(1I64<<b_shift);
            remainder-=delta;
            if(b_shift)
	      {
                --b_shift;
	      }
	  }
      }
    return fixed(internal(),a);
  }

  namespace {
    int const max_power=63-fixed_resolution_shift;
    __int64 const log_two_power_n_reversed[]={
      0x18429946EI64,0x1791272EFI64,0x16DFB516FI64,0x162E42FF0I64,0x157CD0E70I64,0x14CB5ECF1I64,0x1419ECB71I64,0x13687A9F2I64,
      0x12B708872I64,0x1205966F3I64,0x115424573I64,0x10A2B23F4I64,0xFF140274I64,0xF3FCE0F5I64,0xE8E5BF75I64,0xDDCE9DF6I64,
      0xD2B77C76I64,0xC7A05AF7I64,0xBC893977I64,0xB17217F8I64,0xA65AF679I64,0x9B43D4F9I64,0x902CB379I64,0x851591FaI64,
      0x79FE707bI64,0x6EE74EFbI64,0x63D02D7BI64,0x58B90BFcI64,0x4DA1EA7CI64,0x428AC8FdI64,0x3773A77DI64,0x2C5C85FeI64,
      0x2145647EI64,0x162E42FfI64,0xB17217FI64
    };
    
    __int64 const log_one_plus_two_power_minus_n[]={
      0x67CC8FBI64,0x391FEF9I64,0x1E27077I64,0xF85186I64,
      0x7E0A6CI64,0x3F8151I64,0x1FE02AI64,0xFF805I64,0x7FE01I64,0x3FF80I64,0x1FFE0I64,0xFFF8I64,
      0x7FFEI64,0x4000I64,0x2000I64,0x1000I64,0x800I64,0x400I64,0x200I64,0x100I64,
      0x80I64,0x40I64,0x20I64,0x10I64,0x8I64,0x4I64,0x2I64,0x1I64
    };

    __int64 const log_one_over_one_minus_two_power_minus_n[]={
      0xB172180I64,0x49A5884I64,0x222F1D0I64,0x108598BI64,
      0x820AECI64,0x408159I64,0x20202BI64,0x100805I64,0x80201I64,0x40080I64,0x20020I64,0x10008I64,
      0x8002I64,0x4001I64,0x2000I64,0x1000I64,0x800I64,0x400I64,0x200I64,0x100I64,
      0x80I64,0x40I64,0x20I64,0x10I64,0x8I64,0x4I64,0x2I64,0x1I64
    };
  }

  fixed fixed::exp() const {
    if(m_nVal>=log_two_power_n_reversed[0])
      return fixed_max;

    if(m_nVal<-log_two_power_n_reversed[63-2*fixed_resolution_shift])
      return fixed(internal(),0);

    if(!m_nVal)
      return fixed(internal(),fixed_resolution);

    __int64 res=fixed_resolution;

    if(m_nVal>0) {
      int power=max_power;
      __int64 const* log_entry=log_two_power_n_reversed;
      __int64 temp=m_nVal;
      while(temp && power>(-(int)fixed_resolution_shift))
	{
	  while(!power || (temp<*log_entry))
	    {
	      if(!power)
		{
		  log_entry=log_one_plus_two_power_minus_n;
		}
	      else
		{
		  ++log_entry;
		}
	      --power;
	    }
	  temp-=*log_entry;
	  if(power<0)
	    {
	      res+=(res>>(-power));
	    }
	  else
	    {
	      res<<=power;
	    }
	}
    }
    else {
      int power=fixed_resolution_shift;
      __int64 const* log_entry=log_two_power_n_reversed+(max_power-power);
      __int64 temp=m_nVal;

      while(temp && power>(-(int)fixed_resolution_shift)) {
	while(!power || (temp>(-*log_entry))) {
	  if(!power)
	    log_entry=log_one_over_one_minus_two_power_minus_n;

	  else
	    ++log_entry;

	  --power;
	}
	temp+=*log_entry;
	if(power<0)
	  res-=(res>>(-power));

	else
	  res>>=power;
      }
    }
    
    return fixed(internal(),res);
  }

  fixed fixed::log() const {
    if(m_nVal<=0)
      return -fixed_max;

    if(m_nVal==fixed_resolution)
      return fixed_zero;

    unsigned __int64 temp=m_nVal;
    int left_shift=0;
    unsigned __int64 const scale_position=0x8000000000000000;
    while(temp<scale_position) {
      ++left_shift;
      temp<<=1;
    }
    
    __int64 res=(left_shift<max_power)?
      log_two_power_n_reversed[left_shift]:
      -log_two_power_n_reversed[2*max_power-left_shift];
    unsigned right_shift=1;
    unsigned __int64 shifted_temp=temp>>1;
    while(temp && (right_shift<fixed_resolution_shift)) {
      while((right_shift<fixed_resolution_shift) && (temp<(shifted_temp+scale_position))) {
	shifted_temp>>=1;
	++right_shift;
      }
        
      temp-=shifted_temp;
      shifted_temp=temp>>right_shift;
      res+=log_one_over_one_minus_two_power_minus_n[right_shift-1];
    }

    return fixed(fixed::internal(),res);
  }


  namespace {
    const long arctantab[32] = {
      297197971, 210828714, 124459457, 65760959, 33381290, 16755422, 8385879,
      4193963, 2097109, 1048571, 524287, 262144, 131072, 65536, 32768, 16384,
      8192, 4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1, 0, 0,
    };


    long scale_cordic_result(long a) {
      long const cordic_scale_factor=0x22C2DD1C; /* 0.271572 * 2^31*/
      return (long)((((__int64)a)*cordic_scale_factor)>>31);
    }
    
    long right_shift(long val,int shift) {
      return (shift<0)?(val<<-shift):(val>>shift);
    }
    
    void perform_cordic_rotation(long&px, long&py, long theta) {
      long x = px, y = py;
      long const *arctanptr = arctantab;
      for (int i = -1; i <= (int)fixed_resolution_shift; ++i)
	{
	  long const yshift=right_shift(y,i);
	  long const xshift=right_shift(x,i);

	  if (theta < 0)
	    {
	      x += yshift;
	      y -= xshift;
	      theta += *arctanptr++;
	    }
	  else
	    {
	      x -= yshift;
	      y += xshift;
	      theta -= *arctanptr++;
	    }
	}
      px = scale_cordic_result(x);
      py = scale_cordic_result(y);
    }


    void perform_cordic_polarization(long& argx, long&argy) {
      long theta=0;
      long x = argx, y = argy;
      long const *arctanptr = arctantab;
      for(int i = -1; i <= (int)fixed_resolution_shift; ++i)
	{
	  long const yshift=right_shift(y,i);
	  long const xshift=right_shift(x,i);
	  if(y < 0)
	    {
	      y += xshift;
	      x -= yshift;
	      theta -= *arctanptr++;
	    }
	  else
	    {
	      y -= xshift;
	      x += yshift;
	      theta += *arctanptr++;
	    }
	}
      argx = scale_cordic_result(x);
      argy = theta;
    }
  }

  void fixed::sin_cos(fixed const& theta,fixed* s,fixed*c) {
    __int64 x=theta.m_nVal%internal_two_pi;
    if( x < 0 )
      x += internal_two_pi;

    bool negate_cos=false;
    bool negate_sin=false;

    if( x > internal_pi ) {
      x =internal_two_pi-x;
      negate_sin=true;
    }

    if(x>internal_half_pi) {
      x=internal_pi-x;
      negate_cos=true;
    }
    long x_cos=1<<28,x_sin=0;

    perform_cordic_rotation(x_cos,x_sin,(long)x);

    if(s) s->m_nVal=negate_sin?-x_sin:x_sin;

    if(c) c->m_nVal=negate_cos?-x_cos:x_cos;
  }

  fixed fixed::atan() const {
    fixed r,theta;
    to_polar(1,*this,&r,&theta);
    return theta;
  }

  void fixed::to_polar(fixed const& x,fixed const& y,fixed* r,fixed*theta) {
    bool const negative_x=x.m_nVal<0;
    bool const negative_y=y.m_nVal<0;
    
    unsigned __int64 a=negative_x?-x.m_nVal:x.m_nVal;
    unsigned __int64 b=negative_y?-y.m_nVal:y.m_nVal;

    unsigned right_shift=0;
    unsigned const max_value=1U<<fixed_resolution_shift;

    while((a>=max_value) || (b>=max_value))
      {
        ++right_shift;
        a>>=1;
        b>>=1;
      }
    long xtemp=(long)a;
    long ytemp=(long)b;
    perform_cordic_polarization(xtemp,ytemp);
    r->m_nVal=__int64(xtemp)<<right_shift;
    theta->m_nVal=ytemp;

    if(negative_x && negative_y)
      {
        theta->m_nVal-=internal_pi;
      }
    else if(negative_x)
      {
        theta->m_nVal=internal_pi-theta->m_nVal;
      }
    else if(negative_y)
      {
        theta->m_nVal=-theta->m_nVal;
      }
  }



#endif
