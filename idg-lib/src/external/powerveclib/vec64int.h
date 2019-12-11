/******************************************************************************/
/*                                                                            */
/* Licensed Materials - Property of IBM                                       */
/*                                                                            */
/* IBM Power Vector Intrinisic Functions version 1.0.6                        */
/*                                                                            */
/* Copyright IBM Corp. 2015,2017                                              */
/* US Government Users Restricted Rights - Use, duplication or                */
/* disclosure restricted by GSA ADP Schedule Contract with IBM Corp.          */
/*                                                                            */
/* See the licence in the license subdirectory.                               */
/*                                                                            */
/* More information on this software is available on the IBM DeveloperWorks   */
/* website at                                                                 */
/*  https://www.ibm.com/developerworks/community/groups/community/powerveclib */
/*                                                                            */
/******************************************************************************/

#ifndef _H_VEC64INT
#define _H_VEC64INT

#include <altivec.h>
#include "veclib_types.h"

/********************************************************* Set ********************************************************/

/* Set 8 8-bit chars */
VECLIB_INLINE __m64 vec_set8sb (char c7, char c6, char c5, char c4, char c3, char c2, char c1, char c0)
{
  __m64_all_union t;
  #if __LITTLE_ENDIAN__
    t.as_char[0] = c0;
    t.as_char[1] = c1;
    t.as_char[2] = c2;
    t.as_char[3] = c3;
    t.as_char[4] = c4;
    t.as_char[5] = c5;
    t.as_char[6] = c6;
    t.as_char[7] = c7;
  #elif __BIG_ENDIAN__
    t.as_char[0] = c7;
    t.as_char[1] = c6;
    t.as_char[2] = c5;
    t.as_char[3] = c4;
    t.as_char[4] = c3;
    t.as_char[5] = c2;
    t.as_char[6] = c1;
    t.as_char[7] = c0;
  #endif
  return (__m64) t.as_m64;
}

/* Set 4 16-bit shorts */
VECLIB_INLINE __m64 vec_set4sh (short s3, short s2, short s1, short s0)
{
  __m64_all_union t;
  #if __LITTLE_ENDIAN__
    t.as_short[0] = s0;
    t.as_short[1] = s1;
    t.as_short[2] = s2;
    t.as_short[3] = s3;
  #elif __BIG_ENDIAN__
    t.as_short[0] = s3;
    t.as_short[1] = s2;
    t.as_short[2] = s1;
    t.as_short[3] = s0;
  #endif
  return (__m64) t.as_m64;
}

/* Set 4 32-bit ints */
VECLIB_INLINE __m64 vec_set2sw (int i1, int i0)
{
  __m64_union t;
  #ifdef __LITTLE_ENDIAN__
    t.as_int[0] = i0;
    t.as_int[1] = i1;
  #elif __BIG_ENDIAN__
    t.as_int[0] = i1;
    t.as_int[1] = i0;
  #endif
  return (__m64) t.as_m64;
}

/******************************************************* Store ********************************************************/

/* Store long long using a non-temporal memory hint */
VECLIB_INLINE void vec_store1dstream (__m64* to, __m64 from)
{
  #ifdef __ibmxl__
    *to = from;
  /* Non-temporal hint */
    __dcbt ((void *) to);
  #else
    /* Do nothing for now */
  #endif
}

/* Store 8 8-bit chars under mask */
VECLIB_INLINE void vec_storebyvectormask8b (__m64 from, __m64 mask, char* to)
{
  __m64 all_zero = 0x0000000000000000ull;
  __m128_all_union from_union;
  __m128_all_union mask_union;
  __m128_all_union result_union;
  /* Always put __m64 in the left half of VR */
  #ifdef __LITTLE_ENDIAN__
    from_union.as_m64[1] = from; from_union.as_m64[0] = all_zero;
    mask_union.as_m64[1] = mask; mask_union.as_m64[0] = all_zero;
    result_union.as_m64[1] = * ((__m64*) to); result_union.as_m64[0] = all_zero;
  #elif __BIG_ENDIAN__
    from_union.as_m64[0] = from; from_union.as_m64[1] = all_zero;
    mask_union.as_m64[0] = mask; mask_union.as_m64[1] = all_zero;
    result_union.as_m64[0] = * ((__m64*) to); result_union.as_m64[1] = all_zero;
  #endif
  /* Produce the result vector to be stored in to char * to first */
  vector bool char select_vector = vec_cmpgt (mask_union.as_vector_unsigned_char, vec_splats ((unsigned char) 0x80));
  result_union.as_vector_unsigned_char = vec_sel (result_union.as_vector_unsigned_char, from_union.as_vector_unsigned_char, select_vector);

  /* Store the result */
  unsigned long long *to_long = (unsigned long long *) to;
  /* Always put __m64 in the left half of VR */
  #ifdef __LITTLE_ENDIAN__
    *to_long = result_union.as_m64[1];
  #elif __BIG_ENDIAN__
    *to_long = result_union.as_m64[0];
  #endif
}

/******************************************************* Insert *******************************************************/

/* Insert 16-bit short into one of 4 16-bit shorts */
VECLIB_INLINE __m64 vec_insert1hinto4h (__m64 into, int from, intlit2 element_number)
{
  __m64 all_zero = 0x0000000000000000ull;
  vector unsigned short from_vector = vec_splats ((unsigned short) from);
  __m128_all_union into_union;
  __m128_all_union result_union;
  /* Always put __m64 in the left half of VR */
  #ifdef __LITTLE_ENDIAN__
    into_union.as_m64[1] = into; into_union.as_m64[0] = all_zero;
  #elif __BIG_ENDIAN__
    into_union.as_m64[0] = into; into_union.as_m64[1] = all_zero;
  #endif
  static const vector unsigned short select_vectors[4] = {
    #ifdef __LITTLE_ENDIAN__
      {0x0000, 0x0000, 0x0000, 0x0000, 0xFFFF, 0x0000, 0x0000, 0x0000}, /* 0 */
      {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0xFFFF, 0x0000, 0x0000}, /* 1 */
      {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0xFFFF, 0x0000}, /* 2 */
      {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0xFFFF}  /* 3 */
    #elif __BIG_ENDIAN__
      {0x0000, 0x0000, 0x0000, 0xFFFF, 0x0000, 0x0000, 0x0000, 0x0000}, /* 0 */
      {0x0000, 0x0000, 0xFFFF, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}, /* 1 */
      {0x0000, 0xFFFF, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}, /* 2 */
      {0xFFFF, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000}  /* 3 */
    #endif
  };
  result_union.as_vector_unsigned_short = vec_sel (into_union.as_vector_unsigned_short, from_vector, select_vectors[element_number]);
  #ifdef __LITTLE_ENDIAN__
    return result_union.as_m64[1];
  #elif __BIG_ENDIAN__
    return result_union.as_m64[0];
  #endif
}


/***************************************************** Extract ********************************************************/

/* Extract 16-bit short from one of 4 16-bit shorts, zeroing upper */
VECLIB_INLINE int vec_extract1h3zfrom4h (__m64 from, intlit2 element_number)
{
  int result = 0;
  result |= (from >> (element_number * 16)) & 0xFFFF;
  return result;
}

/* Extract upper bit of 8 chars */
VECLIB_INLINE int vec_extractupperbit8b (__m64 from)
{
  __m64_union t;
  t.as_m64 = from;
  int result = 0;
  #ifdef __LITTLE_ENDIAN__
    result |= (t.as_char[7]  & 0x80);
    result |= (t.as_char[6]  & 0x80) >>  (7-6);
    result |= (t.as_char[5]  & 0x80) >>  (7-5);
    result |= (t.as_char[4]  & 0x80) >>  (7-4);
    result |= (t.as_char[3]  & 0x80) >>  (7-3);
    result |= (t.as_char[2]  & 0x80) >>  (7-2);
    result |= (t.as_char[1]  & 0x80) >>  (7-1);
    result |= (t.as_char[0]  & 0x80) >>   7;
  #elif __BIG_ENDIAN__
    result |= (t.as_char[0] & 0x80);
    result |= (t.as_char[1] & 0x80) >>  (7-6);
    result |= (t.as_char[2] & 0x80) >>  (7-5);
    result |= (t.as_char[3] & 0x80) >>  (7-4);
    result |= (t.as_char[4] & 0x80) >>  (7-3);
    result |= (t.as_char[5] & 0x80) >>  (7-2);
    result |= (t.as_char[6] & 0x80) >>  (7-1);
    result |= (t.as_char[7] & 0x80) >>   7;
  #endif
  return result;
}


/*************************************************** Convert Floating-Point to Integer ********************************/

double  __fctiw(double  x);

/* Convert 4 32-bit floats to 8-bit chars and insert */
VECLIB_INLINE __m64 vec_convert4sptolower4of8sb (__m128 a)
{
  __m64_union result;
  __m128_union float_union;
  float_union.as_m128 = a;

  #ifdef __GNUC__
    result.as_signed_char[0] =  __fctiw( (double) float_union.as_float[0]);
    result.as_signed_char[1] =  __fctiw( (double) float_union.as_float[0]);
    result.as_signed_char[2] =  __fctiw( (double) float_union.as_float[0]);;
    result.as_signed_char[3] =  __fctiw( (double) float_union.as_float[0]);
  #elif defined(__ibmxl__)
    double b = __fctiw( (double) float_union.as_float[0]);
    double c = __fctiw( (double) float_union.as_float[1]);
    double d = __fctiw( (double) float_union.as_float[2]);
    double e = __fctiw( (double) float_union.as_float[3]);
    result.as_signed_char[0] = (signed char) *(long long *)&b;
    result.as_signed_char[1] = (signed char) *(long long *)&c;
    result.as_signed_char[2] = (signed char) *(long long *)&d;
    result.as_signed_char[3] = (signed char) *(long long *)&e;
  #endif
  return result.as_m64;
}

/* Convert 4 32-bit floats to 16-bit shorts */
VECLIB_INLINE __m64 vec_convert4spto4sh (__m128 a)
{
  __m64_union result;
  __m128_union float_union;
  float_union.as_m128 = a;
  #ifdef __GNUC__
    result.as_short[0] =  __fctiw( (double) float_union.as_float[0]);
    result.as_short[1] =  __fctiw( (double) float_union.as_float[1]);
    result.as_short[2] =  __fctiw( (double) float_union.as_float[2]);
    result.as_short[3] =  __fctiw( (double) float_union.as_float[3]);
  #elif defined(__ibmxl__)
    double b = __fctiw( (double) float_union.as_float[0]);
    double c = __fctiw( (double) float_union.as_float[1]);
    double d = __fctiw( (double) float_union.as_float[2]);
    double e = __fctiw( (double) float_union.as_float[3]);
    result.as_short[0] = (short) (*(long long *)&b);
    result.as_short[1] = (short) (*(long long *)&c);
    result.as_short[2] = (short) (*(long long *)&d);
    result.as_short[3] = (short) (*(long long *)&e);
  #endif
  return result.as_m64;
}

/* Convert lower 2 32-bit floats to 32-bit ints */
VECLIB_INLINE __m64 vec_convertlower2of4spto2sw (__m128 a)
{
  __m64_union result;
  __m128_union float_union;
  float_union.as_m128 = a;
  #ifdef __GNUC__
    result.as_int[0] =  __fctiw( (double) float_union.as_float[0]);
    result.as_int[1] =  __fctiw( (double) float_union.as_float[1]);
  #elif defined(__ibmxl__)
    double b = __fctiw( (double) float_union.as_float[0]);
    double c = __fctiw( (double) float_union.as_float[1]);
    result.as_int[0] = (int) (*(long long *)&b);
    result.as_int[1] = (int) (*(long long *)&c);
  #endif
  return result.as_m64;
}

/* Convert lower 2 32-bit floats to 32-bit ints with truncation */
VECLIB_INLINE __m64 vec_convertlower2of4spto2swtruncated (__m128 a)
{
  __m64_union result;
  __m128_union float_union;
  float_union.as_m128 = a;
  result.as_int[0] = (int) float_union.as_float[0];
  result.as_int[1] = (int) float_union.as_float[1];
  return result.as_m64;
}

/* Convert lower 32-bit float to 32-bit int */
VECLIB_INLINE int vec_convertlower1of4sptosw (__m128 a)
{
  __m128_union float_union;
  float_union.as_m128 = a;
  #ifdef __GNUC__
    return (int) __fctiw( (double) float_union.as_float[0]);
  #elif defined(__ibmxl__)
    double b = __fctiw( (double) float_union.as_float[0]);
    return (int) (*(long long *)&b);
  #endif
}

/* Convert lower 32-bit float to 32-bit int with truncation */
VECLIB_INLINE int vec_convertlower1of4sptoswtruncated (__m128 a)
{
  __m128_union float_union;
  float_union.as_m128 = a;
  return (int) float_union.as_float[0];

}

/* Convert lower 32-bit float to 64-bit long long */
VECLIB_INLINE long long vec_convertlower1of4spto1sd (__m128 a)
{
  __m128_union float_union;
  float_union.as_m128 = a;
  #ifdef __GNUC__
    return __fctiw( (double) float_union.as_float[0]);
  #elif defined(__ibmxl__)
    double b = __fctiw( (double) float_union.as_float[0]);
    return *(long long *)&b;
  #endif
}


/****************************************************** Boolean *******************************************************/

/* Bitwise 64-bit and */
VECLIB_INLINE __m64 vec_bitwiseand1d (__m64 left, __m64 right)
{
  return (unsigned long long) left & (unsigned long long) right;
}

/* Bitwise 64-bit xor */
VECLIB_INLINE __m64 vec_bitwisexor1d (__m64 left, __m64 right)
{
  return (unsigned long long) left ^ (unsigned long long) right;
}


/****************************************************** Arithmetic ****************************************************/

/* Multiply 4 unsigned 16-bit shorts producing upper halves */
VECLIB_INLINE __m64 vec_multiply4uhupper (__m64 left, __m64 right)
{
  __m64_union left_union; left_union.as_m64 = left;
  __m64_union right_union; right_union.as_m64 = right;
  __m64_union result_union;
  result_union.as_short[0] = (unsigned short) (((unsigned short) left_union.as_short[0] * (unsigned short) right_union.as_short[0]) >> 16);
  result_union.as_short[1] = (unsigned short)(((unsigned short) left_union.as_short[1] * (unsigned short) right_union.as_short[1]) >> 16);
  result_union.as_short[2] = (unsigned short) (((unsigned short) left_union.as_short[2] * (unsigned short) right_union.as_short[2]) >> 16);
  result_union.as_short[3] = (unsigned short) (((unsigned short) left_union.as_short[3] * (unsigned short) right_union.as_short[3]) >> 16);
  return result_union.as_m64;
}

/* Average 8 8-bit unsigned chars */
VECLIB_INLINE __m64 vec_average8ub (__m64 left, __m64 right)
{
  __m64 all_zero = 0x0000000000000000ull;
  __m128_all_union left_union;
  __m128_all_union right_union;
  __m128_all_union result_union;
  /* Always put __m64 in the left half of VR */
  #ifdef __LITTLE_ENDIAN__
    left_union.as_m64[1] = left; left_union.as_m64[0] = all_zero;
    right_union.as_m64[1] = right; right_union.as_m64[0] = all_zero;
  #elif __BIG_ENDIAN__
    left_union.as_m64[0] = left; left_union.as_m64[1] = all_zero;
    right_union.as_m64[0] = right; right_union.as_m64[1] = all_zero;
  #endif
  result_union.as_vector_unsigned_char = vec_avg (left_union.as_vector_unsigned_char, right_union.as_vector_unsigned_char);
  #ifdef __LITTLE_ENDIAN__
    return result_union.as_m64[1];
  #elif __BIG_ENDIAN__
    return result_union.as_m64[0];
  #endif
}

/* Average 4 16-bit unsigned shorts */
VECLIB_INLINE __m64 vec_average4uh (__m64 left, __m64 right)
{
  __m64 all_zero = 0x0000000000000000ull;
  __m128_all_union left_union;
  __m128_all_union right_union;
  __m128_all_union result_union;
  /* Always put __m64 in the left half of VR */
  #ifdef __LITTLE_ENDIAN__
    left_union.as_m64[1] = left; left_union.as_m64[0] = all_zero;
    right_union.as_m64[1] = right; right_union.as_m64[0] = all_zero;
  #elif __BIG_ENDIAN__
    left_union.as_m64[0] = left; left_union.as_m64[1] = all_zero;
    right_union.as_m64[0] = right; right_union.as_m64[1] = all_zero;
  #endif
  result_union.as_vector_unsigned_short = vec_avg (left_union.as_vector_unsigned_short, right_union.as_vector_unsigned_short);
  #ifdef __LITTLE_ENDIAN__
    return result_union.as_m64[1];
  #elif __BIG_ENDIAN__
    return result_union.as_m64[0];
  #endif
}

/* Max 8 8-bit unsigned chars */
VECLIB_INLINE __m64 vec_max8ub (__m64 left, __m64 right)
{
  __m64 all_zero = 0x0000000000000000ull;
  __m128_all_union left_union;
  __m128_all_union right_union;
  __m128_all_union result_union;
  /* Always put __m64 in the left half of VR */
  #ifdef __LITTLE_ENDIAN__
    left_union.as_m64[1] = left; left_union.as_m64[0] = all_zero;
    right_union.as_m64[1] = right; right_union.as_m64[0] = all_zero;
  #elif __BIG_ENDIAN__
    left_union.as_m64[0] = left; left_union.as_m64[1] = all_zero;
    right_union.as_m64[0] = right; right_union.as_m64[1] = all_zero;
  #endif
  result_union.as_vector_unsigned_char = vec_max (left_union.as_vector_unsigned_char, right_union.as_vector_unsigned_char);
  #ifdef __LITTLE_ENDIAN__
    return result_union.as_m64[1];
  #elif __BIG_ENDIAN__
    return result_union.as_m64[0];
  #endif
}

/* Max 4 16-bit shorts */
VECLIB_INLINE __m64 vec_max4sh (__m64 left, __m64 right)
{
  __m64 all_zero = 0x0000000000000000ull;
  __m128_all_union left_union;
  __m128_all_union right_union;
  __m128_all_union result_union;
  /* Always put __m64 in the left half of VR */
  #ifdef __LITTLE_ENDIAN__
    left_union.as_m64[1] = left; left_union.as_m64[0] = all_zero;
    right_union.as_m64[1] = right; right_union.as_m64[0] = all_zero;
  #elif __BIG_ENDIAN__
    left_union.as_m64[0] = left; left_union.as_m64[1] = all_zero;
    right_union.as_m64[0] = right; right_union.as_m64[1] = all_zero;
  #endif
  result_union.as_vector_signed_short = vec_max (left_union.as_vector_signed_short, right_union.as_vector_signed_short);
  #ifdef __LITTLE_ENDIAN__
    return result_union.as_m64[1];
  #elif __BIG_ENDIAN__
    return result_union.as_m64[0];
  #endif
}

/* Min 8 8-bit unsigned chars */
VECLIB_INLINE __m64 vec_min8ub (__m64 left, __m64 right)
{
  __m64 all_zero = 0x0000000000000000ull;
  __m128_all_union left_union;
  __m128_all_union right_union;
  __m128_all_union result_union;
  /* Always put __m64 in the left half of VR */
  #ifdef __LITTLE_ENDIAN__
    left_union.as_m64[1] = left; left_union.as_m64[0] = all_zero;
    right_union.as_m64[1] = right; right_union.as_m64[0] = all_zero;
  #elif __BIG_ENDIAN__
    left_union.as_m64[0] = left; left_union.as_m64[1] = all_zero;
    right_union.as_m64[0] = right; right_union.as_m64[1] = all_zero;
  #endif
  result_union.as_vector_unsigned_char = vec_min (left_union.as_vector_unsigned_char, right_union.as_vector_unsigned_char);
  #ifdef __LITTLE_ENDIAN__
    return result_union.as_m64[1];
  #elif __BIG_ENDIAN__
    return result_union.as_m64[0];
  #endif
}

/* Min 4 16-bit shorts */
VECLIB_INLINE __m64 vec_min4sh (__m64 left, __m64 right)
{
  __m64 all_zero = 0x0000000000000000ull;
  __m128_all_union left_union;
  __m128_all_union right_union;
  __m128_all_union result_union;
  /* Always put __m64 in the left half of VR */
  #ifdef __LITTLE_ENDIAN__
    left_union.as_m64[1] = left; left_union.as_m64[0] = all_zero;
    right_union.as_m64[1] = right; right_union.as_m64[0] = all_zero;
  #elif __BIG_ENDIAN__
    left_union.as_m64[0] = left; left_union.as_m64[1] = all_zero;
    right_union.as_m64[0] = right; right_union.as_m64[1] = all_zero;
  #endif
  result_union.as_vector_signed_short = vec_min (left_union.as_vector_signed_short, right_union.as_vector_signed_short);
  #ifdef __LITTLE_ENDIAN__
    return result_union.as_m64[1];
  #elif __BIG_ENDIAN__
    return result_union.as_m64[0];
  #endif
}

/* Sum absolute differences of 8 8-bit unsigned chars */
VECLIB_INLINE __m64 vec_sumabsdiffs8ub (__m64 left, __m64 right)
{
  __m64 all_zero = 0x0000000000000000ull;
  __m128_all_union left_union;
  __m128_all_union right_union;
  __m128_all_union result_union;
  /* Always put __m64 in the left half of VR */
  #ifdef __LITTLE_ENDIAN__
    left_union.as_m64[1] = left; left_union.as_m64[0] = all_zero;
    right_union.as_m64[1] = right; right_union.as_m64[0] = all_zero;
  #elif __BIG_ENDIAN__
    left_union.as_m64[0] = left; left_union.as_m64[1] = all_zero;
    right_union.as_m64[0] = right; right_union.as_m64[1] = all_zero;
  #endif

  vector unsigned char minimums = vec_min (left_union.as_vector_unsigned_char, right_union.as_vector_unsigned_char);
  vector unsigned char maximums = vec_max (left_union.as_vector_unsigned_char, right_union.as_vector_unsigned_char);
  vector unsigned char absolute_differences = vec_sub (maximums, minimums);
  vector unsigned int int_sums =  vec_sum4s (absolute_differences, vec_splats ((unsigned int) 0u));
  int_sums = (vector unsigned int) vec_sums ((vector signed int) int_sums, vec_splats ((signed int) 0));
  #ifdef __ibmxl__
    #ifdef __LITTLE_ENDIAN__
      vector unsigned char permute_vector = {
        0x1C, 0x1D, 0x1E, 0x1F,  0x18, 0x19, 0x1A, 0x1B,  0x14, 0x15, 0x16, 0x17, 0x10, 0x11, 0x12, 0x13
      };
      int_sums = vec_perm (int_sums, int_sums, permute_vector);
      result_union.as_vector_unsigned_int = int_sums;
      return result_union.as_m64[0];
    #elif __BIG_ENDIAN__
      result_union.as_vector_unsigned_int = int_sums;
      return result_union.as_m64[1];
    #endif
  #elif __GNUC__
    __m64_union t;
    result_union.as_vector_unsigned_int = int_sums;
    #ifdef __LITTLE_ENDIAN__
      t.as_short[0] = result_union.as_int[0] + result_union.as_int[1] + result_union.as_int[2] + result_union.as_int[3];
      t.as_short[1] = 0;
      t.as_short[2] = 0;
      t.as_short[3] = 0;
      return t.as_m64;
    #elif __BIG_ENDIAN__
      t.as_short[3] = result_union.as_int[0] + result_union.as_int[1] + result_union.as_int[2] + result_union.as_int[3];
      t.as_short[2] = 0;
      t.as_short[1] = 0;
      t.as_short[0] = 0;
      return t.as_m64;
    #endif
  #endif
}

/* Absolute value 8 8-bit chars */
VECLIB_INLINE __m64 vec_Abs8sb (__m64 v) {
  __m128_all_union temp; temp.as_m64[0] = v;
  temp.as_vector_signed_char = vec_abs(temp.as_vector_signed_char);
  return temp.as_m64[0];
}

/* Absolute value 4 16-bit shorts */
VECLIB_INLINE __m64 vec_Abs4sh (__m64 v) {
  __m128_all_union temp; temp.as_m64[0] = v;
  temp.as_vector_signed_short = vec_abs(temp.as_vector_signed_short);
  return temp.as_m64[0];
}

/* Absolute value 2 32-bit ints */
VECLIB_INLINE __m64 vec_Abs2sw (__m64 v) {
  __m128_all_union temp; temp.as_m64[0] = v;
  temp.as_vector_signed_int = vec_abs(temp.as_vector_signed_int);
  return temp.as_m64[0];
}

/* Horizontally add 2+2 adjacent pairs of 16-bit shorts to 4 16-bit shorts - (a0+a1, a2+a3, b0+b1, b2+b3) */
VECLIB_INLINE __m64 vec_horizontalAdd4sh (__m64 left, __m64 right) {
  __m128_all_union temp;  temp.as_m64[0] = left;  temp.as_m64[1] = right;
  #ifdef __LITTLE_ENDIAN__
    static vector unsigned char evens = (vector unsigned char) {
      0x00,0x01,0x04,0x05,0x08,0x09,0x0C,0x0D,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    static vector unsigned char odds = (vector unsigned char) {
      0x02,0x03,0x06,0x07,0x0A,0x0B,0x0E,0x0F,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
  #elif __BIG_ENDIAN__
    static vector unsigned char evens = (vector unsigned char) {
      0x08,0x09,0x0C,0x0D,0x00,0x01,0x04,0x05,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    static vector unsigned char odds = (vector unsigned char) {
      0x0A,0x0B,0x0E,0x0F,0x02,0x03,0x06,0x07,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
  #endif
  temp.as_vector_signed_short = vec_add((vector signed short)vec_perm(temp.as_m128, temp.as_m128, evens), (vector signed short)vec_perm(temp.as_m128, temp.as_m128, odds));
  return temp.as_m64[0];
}

/* Horizontally add 1+1 adjacent pairs of 32-bit ints to 2 32-bit ints - (a0+a1, b0+b1) */
VECLIB_INLINE __m64 vec_partialhorizontaladd1sw (__m64 left, __m64 right) {
  __m128_all_union temp;  temp.as_m64[0] = left;  temp.as_m64[1] = right;
  #ifdef __LITTLE_ENDIAN__
    static vector unsigned char evens = (vector unsigned char) {
      0x00,0x01,0x02,0x03,0x08,0x09,0x0A,0x0B,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    static vector unsigned char odds = (vector unsigned char) {
      0x04,0x05,0x06,0x07,0x0C,0x0D,0x0E,0x0F,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
  #elif __BIG_ENDIAN__
    static vector unsigned char evens = (vector unsigned char) {
      0x08,0x09,0x0A,0x0B,0x00,0x01,0x02,0x03,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    static vector unsigned char odds = (vector unsigned char) {
      0x0C,0x0D,0x0E,0x0F,0x04,0x05,0x06,0x07,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
  #endif
  temp.as_vector_signed_int = vec_add((vector signed int)vec_perm(temp.as_m128, temp.as_m128, evens), (vector signed int)vec_perm(temp.as_m128, temp.as_m128, odds));
  return temp.as_m64[0];
}

/* Horizontally add 2+2 adjacent pairs of 16-bit shorts to 4 16-bit shorts with saturation - (a0+a1, a2+a3, b0+b1, b2+b3) */
VECLIB_INLINE __m64 vec_horizontalAddsaturating4sh (__m64 left, __m64 right) {
  __m128_all_union temp;  temp.as_m64[0] = left;  temp.as_m64[1] = right;
  #ifdef __LITTLE_ENDIAN__
    static vector unsigned char evens = (vector unsigned char) {
      0x00,0x01,0x04,0x05,0x08,0x09,0x0C,0x0D,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    static vector unsigned char odds = (vector unsigned char) {
      0x02,0x03,0x06,0x07,0x0A,0x0B,0x0E,0x0F,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
  #elif __BIG_ENDIAN__
    static vector unsigned char evens = (vector unsigned char) {
      0x08,0x09,0x0C,0x0D,0x00,0x01,0x04,0x05,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    static vector unsigned char odds = (vector unsigned char) {
      0x0A,0x0B,0x0E,0x0F,0x02,0x03,0x06,0x07,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
  #endif
  temp.as_vector_signed_short = vec_adds((vector signed short)vec_perm(temp.as_m128, temp.as_m128, evens), (vector signed short)vec_perm(temp.as_m128, temp.as_m128, odds));
  return temp.as_m64[0];
}

/* Horizontally subtract 2+2 adjacent pairs of 16-bit shorts to 4 16-bit shorts - (a0+a1, a2+a3, b0+b1, b2+b3) */
VECLIB_INLINE __m64 vec_horizontalSub4sh (__m64 left, __m64 right) {
  __m128_all_union temp;  temp.as_m64[0] = left;  temp.as_m64[1] = right;

  #ifdef __LITTLE_ENDIAN__
    static vector unsigned char evens = (vector unsigned char) {
      0x00,0x01,0x04,0x05,0x08,0x09,0x0C,0x0D,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    static vector unsigned char odds = (vector unsigned char) {
      0x02,0x03,0x06,0x07,0x0A,0x0B,0x0E,0x0F,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
  #elif __BIG_ENDIAN__
    static vector unsigned char evens = (vector unsigned char) {
      0x0A,0x0B,0x0E,0x0F,0x02,0x03,0x06,0x07,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    static vector unsigned char odds = (vector unsigned char) {
      0x08,0x09,0x0C,0x0D,0x00,0x01,0x04,0x05,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
  #endif
  temp.as_vector_signed_short = vec_sub((vector signed short)vec_perm(temp.as_m128, temp.as_m128, evens), (vector signed short)vec_perm(temp.as_m128, temp.as_m128, odds));
  return temp.as_m64[0];
}

/* Horizontally subtract 1+1 adjacent pairs of 32-bit ints to 2 32-bit ints - (a0+a1, b0+b1) */
VECLIB_INLINE __m64 vec_partialhorizontalsub1sw (__m64 left, __m64 right) {
  __m128_all_union temp;  temp.as_m64[0] = left;  temp.as_m64[1] = right;
  #ifdef __LITTLE_ENDIAN__
    static vector unsigned char evens = (vector unsigned char) {
      0x00,0x01,0x02,0x03,0x08,0x09,0x0A,0x0B,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    static vector unsigned char odds = (vector unsigned char) {
      0x04,0x05,0x06,0x07,0x0C,0x0D,0x0E,0x0F,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
  #elif __BIG_ENDIAN__
    static vector unsigned char evens = (vector unsigned char) {
      0x0C,0x0D,0x0E,0x0F,0x04,0x05,0x06,0x07,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    static vector unsigned char odds = (vector unsigned char) {
      0x08,0x09,0x0A,0x0B,0x00,0x01,0x02,0x03,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
  #endif
  temp.as_vector_signed_int = vec_sub((vector signed int)vec_perm(temp.as_m128, temp.as_m128, evens), (vector signed int)vec_perm(temp.as_m128, temp.as_m128, odds));
  return temp.as_m64[0];
}

/* Horizontally subtract 2+2 adjacent pairs of 16-bit shorts to 4 16-bit shorts with saturation - (a0+a1, a2+a3, b0+b1, b2+b3) */
VECLIB_INLINE __m64 vec_horizontalSubsaturating4sh (__m64 left, __m64 right) {
  __m128_all_union temp;  temp.as_m64[0] = left;  temp.as_m64[1] = right;
  #ifdef __LITTLE_ENDIAN__
    static vector unsigned char evens = (vector unsigned char) {
      0x00,0x01,0x04,0x05,0x08,0x09,0x0C,0x0D,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    static vector unsigned char odds = (vector unsigned char) {
      0x02,0x03,0x06,0x07,0x0A,0x0B,0x0E,0x0F,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
  #elif __BIG_ENDIAN__
    static vector unsigned char evens = (vector unsigned char) {
      0x0A,0x0B,0x0E,0x0F,0x02,0x03,0x06,0x07,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
    static vector unsigned char odds = (vector unsigned char) {
      0x08,0x09,0x0C,0x0D,0x00,0x01,0x04,0x05,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00};
  #endif
  temp.as_vector_signed_short = vec_subs((vector signed short)vec_perm(temp.as_m128, temp.as_m128, evens), (vector signed short)vec_perm(temp.as_m128, temp.as_m128, odds));
  return temp.as_m64[0];
}

/* Multiply 8 8-bit signed chars then add adjacent 16-bit products with signed saturation */
VECLIB_INLINE __m64 vec_Multiply8sbthenhorizontalAddsaturating8sh (__m64 left, __m64 right) {
  __m128i_union inputs; inputs.as_vector_unsigned_long_long = (vector unsigned long long) { left, right};

  vector signed short zeroUpperHalfOfInts = { 0x00FF, 0x00FF, 0x00FF, 0x00FF, 0x00FF, 0x00FF, 0x00FF, 0x00FF };
  /* This extends the "left" to be 16-bit unsigned integer version of iteself. The vec_and prevents the makes simulates that the value is unsigned */
  __m128i_union unionLeft;  unionLeft.as_vector_signed_short =  vec_and( vec_unpackh( inputs.as_vector_signed_char ), zeroUpperHalfOfInts );
  /* This extends the "Right" to 16-bit signed integer version of itself. The unpack will extend the sign to the entire integer */
  __m128i_union unionRight; unionRight.as_vector_signed_short = vec_unpackl( inputs.as_vector_signed_char );
  __m128i_union unionResult;
  unionResult.as_vector_signed_short = vec_packs (
    vec_add(
      vec_mulo(unionLeft.as_vector_signed_short, unionRight.as_vector_signed_short),
      vec_mule(unionLeft.as_vector_signed_short, unionRight.as_vector_signed_short)
    ),
    unionLeft.as_vector_signed_int /* this part will be discarded */
  );
  return unionResult.as_m64[0];
}

/* Multiply 4 16-bit shorts, shift right 14, add 1 and shift right 1 to 4 16-bit shorts */
VECLIB_INLINE __m64 vec_Multiply4shExtractUpper (__m64 left, __m64 right) {

  __m128i_union inputs; inputs.as_vector_unsigned_long_long = (vector unsigned long long) {left, right};

  #ifdef __LITTLE_ENDIAN__
    inputs.as_vector_signed_int = vec_mule( (vector signed short)vec_unpackh(inputs.as_vector_signed_short), (vector signed short)vec_unpackl(inputs.as_vector_signed_short) );
  #elif __BIG_ENDIAN__
    inputs.as_vector_signed_int = vec_mulo( (vector signed short)vec_unpackh(inputs.as_vector_signed_short), (vector signed short)vec_unpackl(inputs.as_vector_signed_short) );
  #endif

  static vector unsigned int addVector   = (vector unsigned int) { 0x00004000u, 0x00004000u, 0x00004000u, 0x00004000u };
  static vector unsigned int shiftVector = (vector unsigned int) { 0x01010101u, 0x01010101u, 0x01010101u, 0x01010101u };
  inputs.as_vector_unsigned_int = vec_sll(vec_add(inputs.as_vector_unsigned_int, addVector), shiftVector);

  static vector unsigned char permuteMask = (vector unsigned char) {
    #ifdef __LITTLE_ENDIAN__
      0x02, 0x03, 0x06, 0x07,  0x0A, 0x0B, 0x0E, 0x0F,  0x10, 0x10, 0x10, 0x10,  0x10, 0x10, 0x10, 0x10,
    #elif __BIG_ENDIAN__
      0x00, 0x01, 0x04, 0x05,  0x08, 0x09, 0x0C, 0x0D,  0x10, 0x10, 0x10, 0x10,  0x10, 0x10, 0x10, 0x10,  //0x00, 0x01, 0x04, 0x05,  0x08, 0x09, 0x0C, 0x0D,
    #endif
  };
  inputs.as_m128i = (__m128i)vec_perm(inputs.as_vector_unsigned_int, addVector, permuteMask); //addVector does not do anything, I was forced to put a parameter there for the vecperm
  return inputs.as_m64[0];
}

/* Negate 8 8-bit chars when mask is negative, zero when zero, else copy */
VECLIB_INLINE __m64 vec_conditionalNegate8sb (__m64 left, __m64 right) {
  __m128i_union newLeft; newLeft.as_m64[0]= left;
  __m128i_union newRight; newRight.as_m64[0]= right;
  __m128i_union returnedResult;
  vector signed char zeroes = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  returnedResult.as_vector_signed_char = vec_sel(
  vec_sel(
    (vector signed char)vec_sub(zeroes, (vector signed char)newLeft.as_vector_signed_char),
    (vector signed char)newLeft.as_vector_signed_char,
    vec_cmpgt((vector signed char)newRight.as_vector_signed_char, zeroes)),
  zeroes,
  vec_cmpeq((vector signed char)newRight.as_vector_signed_char, zeroes));
  return returnedResult.as_m64[0];
}

/* Negate 4 16-bit shorts when mask is negative, zero when zero, else copy */
VECLIB_INLINE __m64 vec_conditionalNegate4sh (__m64 left, __m64 right) {
  __m128i_union newLeft; newLeft.as_m64[0]= left;
  __m128i_union newRight; newRight.as_m64[0]= right;
  __m128i_union returnedResult;
  vector signed short zeroes = {0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, };
  returnedResult.as_vector_signed_short = vec_sel(
  vec_sel(
    (vector signed short)vec_sub(zeroes, (vector signed short)newLeft.as_vector_signed_short),
    (vector signed short)newLeft.as_vector_signed_short,
    vec_cmpgt((vector signed short)newRight.as_vector_signed_short, zeroes)),
  zeroes,
  vec_cmpeq((vector signed short)newRight.as_vector_signed_short, zeroes));
  return returnedResult.as_m64[0];
}

/* Negate 2 32-bit ints when mask is negative, zero when zero, else copy */
VECLIB_INLINE __m64 vec_conditionalNegate2sw (__m64 left, __m64 right) {
  __m128i_union newLeft; newLeft.as_m64[0]= left;
  __m128i_union newRight; newRight.as_m64[0]= right;
  __m128i_union returnedResult;
  vector signed int zeroes = {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u };
  returnedResult.as_vector_signed_int = vec_sel(
  vec_sel(
    (vector signed int)vec_sub(zeroes, (vector signed int)newLeft.as_vector_signed_int),
    (vector signed int)newLeft.as_vector_signed_int,
    vec_cmpgt((vector signed int)newRight.as_vector_signed_int, zeroes)),
  zeroes,
  vec_cmpeq((vector signed int)newRight.as_vector_signed_int, zeroes));
  return returnedResult.as_m64[0];
}

/* Add 4 signed 16-bit shorts */
VECLIB_INLINE __m64 vec_add4sh (__m64 left, __m64 right) {
  __m128_all_union temp_left; temp_left.as_m64[0] = left;
  __m128_all_union temp_right; temp_right.as_m64[0] = right;
  __m128_all_union temp;
  temp.as_vector_signed_short = vec_add(temp_left.as_vector_signed_short, temp_right.as_vector_signed_short);
  return temp.as_m64[0];
}

/* Add 2 32-bit ints */
VECLIB_INLINE __m64 vec_add2sw (__m64 left, __m64 right) {
  __m128_all_union temp_left; temp_left.as_m64[0] = left;
  __m128_all_union temp_right; temp_right.as_m64[0] = right;
  __m128_all_union temp;
  temp.as_vector_signed_int = vec_add(temp_left.as_vector_signed_int, temp_right.as_vector_signed_int);
  return temp.as_m64[0];
}

/* Add 8 8-bit signed chars */
VECLIB_INLINE __m64 vec_add8sb (__m64 left, __m64 right) {
  __m128_all_union temp_left; temp_left.as_m64[0] = left;
  __m128_all_union temp_right; temp_right.as_m64[0] = right;
  __m128_all_union temp;
  temp.as_vector_signed_char = vec_add(temp_left.as_vector_signed_char, temp_right.as_vector_signed_char);
  return temp.as_m64[0];
}

/* Add 4 signed 16-bit shorts with saturation */
VECLIB_INLINE __m64 vec_addsaturating4sh (__m64 left, __m64 right) {
  __m128_all_union temp_left; temp_left.as_m64[0] = left;
  __m128_all_union temp_right; temp_right.as_m64[0] = right;
  __m128_all_union temp;
  temp.as_vector_signed_short = vec_adds(temp_left.as_vector_signed_short, temp_right.as_vector_signed_short);
  return temp.as_m64[0];
}

/* Add 8 8-bit signed chars with saturation */
VECLIB_INLINE __m64 vec_addsaturating8sb (__m64 left, __m64 right) {
  __m128_all_union temp_left; temp_left.as_m64[0] = left;
  __m128_all_union temp_right; temp_right.as_m64[0] = right;
  __m128_all_union temp;
  temp.as_vector_signed_char = vec_adds(temp_left.as_vector_signed_char, temp_right.as_vector_signed_char);
  return temp.as_m64[0];
}

/* Add 4 unsigned 16-bit shorts with saturation */
VECLIB_INLINE __m64 vec_addsaturating4uh (__m64 left, __m64 right) {
  __m128_all_union temp_left; temp_left.as_m64[0] = left;
  __m128_all_union temp_right; temp_right.as_m64[0] = right;
  __m128_all_union temp;
  temp.as_vector_unsigned_short = vec_adds(temp_left.as_vector_unsigned_short, temp_right.as_vector_unsigned_short);
  return temp.as_m64[0];
}

/* Add 8 8-bit unsigned chars with saturation */
VECLIB_INLINE __m64 vec_addsaturating8ub (__m64 left, __m64 right) {
  __m128_all_union temp_left; temp_left.as_m64[0] = left;
  __m128_all_union temp_right; temp_right.as_m64[0] = right;
  __m128_all_union temp;
  temp.as_vector_unsigned_char = vec_adds(temp_left.as_vector_unsigned_char, temp_right.as_vector_unsigned_char);
  return temp.as_m64[0];
}


/* Subtract 4 signed 16-bit shorts */
VECLIB_INLINE __m64 vec_subtract4sh (__m64 left, __m64 right) {
  __m128_all_union temp_left; temp_left.as_m64[0] = left;
  __m128_all_union temp_right; temp_right.as_m64[0] = right;
  __m128_all_union temp;
  temp.as_vector_signed_short = vec_sub(temp_left.as_vector_signed_short, temp_right.as_vector_signed_short);
  return temp.as_m64[0];
}

/* Subtract 2 32-bit ints */
VECLIB_INLINE __m64 vec_subtract2sw (__m64 left, __m64 right) {
  __m128_all_union temp_left; temp_left.as_m64[0] = left;
  __m128_all_union temp_right; temp_right.as_m64[0] = right;
  __m128_all_union temp;
  temp.as_vector_signed_int = vec_sub(temp_left.as_vector_signed_int, temp_right.as_vector_signed_int);
  return temp.as_m64[0];
}

/* Subtract 8 8-bit signed chars */
VECLIB_INLINE __m64 vec_subtract8sb (__m64 left, __m64 right) {
  __m128_all_union temp_left; temp_left.as_m64[0] = left;
  __m128_all_union temp_right; temp_right.as_m64[0] = right;
  __m128_all_union temp;
  temp.as_vector_signed_char = vec_sub(temp_left.as_vector_signed_char, temp_right.as_vector_signed_char);
  return temp.as_m64[0];
}

/* Subtract 4 signed 16-bit shorts with saturation */
VECLIB_INLINE __m64 vec_subtractsaturating4sh (__m64 left, __m64 right) {
  __m128_all_union temp_left; temp_left.as_m64[0] = left;
  __m128_all_union temp_right; temp_right.as_m64[0] = right;
  __m128_all_union temp;
  temp.as_vector_signed_short = vec_subs(temp_left.as_vector_signed_short, temp_right.as_vector_signed_short);
  return temp.as_m64[0];
}

/* Subtract 8 8-bit signed chars with saturation */
VECLIB_INLINE __m64 vec_subtractsaturating8sb (__m64 left, __m64 right) {
  __m128_all_union temp_left; temp_left.as_m64[0] = left;
  __m128_all_union temp_right; temp_right.as_m64[0] = right;
  __m128_all_union temp;
  temp.as_vector_signed_char = vec_subs(temp_left.as_vector_signed_char, temp_right.as_vector_signed_char);
  return temp.as_m64[0];
}

/* Subtract 4 unsigned 16-bit shorts with saturation */
VECLIB_INLINE __m64 vec_subtractsaturating4uh (__m64 left, __m64 right) {
  __m128_all_union temp_left; temp_left.as_m64[0] = left;
  __m128_all_union temp_right; temp_right.as_m64[0] = right;
  __m128_all_union temp;
  temp.as_vector_unsigned_short = vec_subs(temp_left.as_vector_unsigned_short, temp_right.as_vector_unsigned_short);
  return temp.as_m64[0];
}

/* Subtract 8 8-bit unsigned chars with saturation */
VECLIB_INLINE __m64 vec_subtractsaturating8ub (__m64 left, __m64 right) {
  __m128_all_union temp_left; temp_left.as_m64[0] = left;
  __m128_all_union temp_right; temp_right.as_m64[0] = right;
  __m128_all_union temp;
  temp.as_vector_unsigned_char = vec_subs(temp_left.as_vector_unsigned_char, temp_right.as_vector_unsigned_char);
  return temp.as_m64[0];
}


/******************************************************* Comparison ********************************************************/


/* Compare 4 signed 16-bit shorts for == to mask */
VECLIB_INLINE __m64 vec_compareeq4sh (__m64 left, __m64 right) {
  __m128_all_union temp_left; temp_left.as_m64[0] = left;
  __m128_all_union temp_right; temp_right.as_m64[0] = right;
  __m128_all_union temp;
  temp.as_vector_bool_short = vec_cmpeq(temp_left.as_vector_signed_short, temp_right.as_vector_signed_short);
  return temp.as_m64[0];
}

/* Compare 2 32-bit ints for == to mask */
VECLIB_INLINE __m64 vec_compareeq2sw (__m64 left, __m64 right) {
  __m128_all_union temp_left; temp_left.as_m64[0] = left;
  __m128_all_union temp_right; temp_right.as_m64[0] = right;
  __m128_all_union temp;
  temp.as_vector_bool_int = vec_cmpeq(temp_left.as_vector_signed_int, temp_right.as_vector_signed_int);
  return temp.as_m64[0];
}

/* Compare 8 8-bit signed chars for == to mask */
VECLIB_INLINE __m64 vec_compareeq8sb (__m64 left, __m64 right) {
  __m128_all_union temp_left; temp_left.as_m64[0] = left;
  __m128_all_union temp_right; temp_right.as_m64[0] = right;
  __m128_all_union temp;
  temp.as_vector_bool_char = vec_cmpeq(temp_left.as_vector_signed_char, temp_right.as_vector_signed_char);
  return temp.as_m64[0];
}

/* Compare 4 signed 16-bit shorts for > to mask */
VECLIB_INLINE __m64 vec_comparegt4sh (__m64 left, __m64 right) {
  __m128_all_union temp_left; temp_left.as_m64[0] = left;
  __m128_all_union temp_right; temp_right.as_m64[0] = right;
  __m128_all_union temp;
  temp.as_vector_bool_short = vec_cmpgt(temp_left.as_vector_signed_short, temp_right.as_vector_signed_short);
  return temp.as_m64[0];
}

/* Compare 2 32-bit ints for > to mask */
VECLIB_INLINE __m64 vec_comparegt2sw (__m64 left, __m64 right) {
  __m128_all_union temp_left; temp_left.as_m64[0] = left;
  __m128_all_union temp_right; temp_right.as_m64[0] = right;
  __m128_all_union temp;
  temp.as_vector_bool_int = vec_cmpgt(temp_left.as_vector_signed_int, temp_right.as_vector_signed_int);
  return temp.as_m64[0];
}

/* Compare 8 8-bit signed chars for > to mask */
VECLIB_INLINE __m64 vec_comparegt8sb (__m64 left, __m64 right) {
  __m128_all_union temp_left; temp_left.as_m64[0] = left;
  __m128_all_union temp_right; temp_right.as_m64[0] = right;
  __m128_all_union temp;
  temp.as_vector_bool_char = vec_cmpgt(temp_left.as_vector_signed_char, temp_right.as_vector_signed_char);
  return temp.as_m64[0];
}

/******************************************************* Shift ********************************************************/

/* Shift 64-bit long long right logical */
VECLIB_INLINE __m64 vec_shiftrightlogical1dimmediate (__m64 v, intlit8 count)
{
  if ((unsigned long) count >= 64)
  {
    return 0x0000000000000000ull;
  } else if (count == 0) {
    return v;
  } else {
    return (unsigned long long) v >> count;
  }
}

/* Shift 64+64-bits right into 64-bits */
VECLIB_INLINE __m64 vec_shiftright2dw (__m64 left, __m64 right, int count) {
  if (count < 8) {
    return (right >> (count*8)) | (left << ((8-count)*8));
  }
  else if(count < 16) {
    return (left >> ((count - 8)*8));
  }
  else{
    return (__m64)0;
  }
}

/******************************************************* Permute ******************************************************/

/* Shuffle 4 16-bit shorts */
VECLIB_INLINE __m64 vec_permute4himmediate (__m64 from, intlit8 selector)
{
  __m64 all_zero = 0x0000000000000000ull;
  __m128_all_union from_union;
  __m128_all_union result_union;
  /* Always put __m64 in the left half of VR */
  #ifdef __LITTLE_ENDIAN__
    from_union.as_m64[1] = from; from_union.as_m64[0] = all_zero;
  #elif __BIG_ENDIAN__
    from_union.as_m64[0] = from; from_union.as_m64[1] = all_zero;
  #endif
  unsigned long element_selector_10 =  selector       & 0x03;
  unsigned long element_selector_32 = (selector >> 2) & 0x03;
  unsigned long element_selector_54 = (selector >> 4) & 0x03;
  unsigned long element_selector_76 = (selector >> 6) & 0x03;
  const static unsigned short permute_selectors[4] = {
    #ifdef __LITTLE_ENDIAN__
      0x0908, 0x0B0A, 0x0D0C, 0x0F0E
    #elif __BIG_ENDIAN__
      0x0607, 0x0405, 0x0203, 0x0001
    #endif
  };
  __m128i_union permute_selector;
  #ifdef __LITTLE_ENDIAN__
    permute_selector.as_m64[0] = 0x0706050403020100ull;
    permute_selector.as_short[4] = permute_selectors [element_selector_10];
    permute_selector.as_short[5] = permute_selectors [element_selector_32];
    permute_selector.as_short[6] = permute_selectors[element_selector_54];
    permute_selector.as_short[7] = permute_selectors[element_selector_76];
  #elif __BIG_ENDIAN__
    permute_selector.as_short[0] = permute_selectors[element_selector_76];
    permute_selector.as_short[1] = permute_selectors[element_selector_54];
    permute_selector.as_short[2] = permute_selectors [element_selector_32];
    permute_selector.as_short[3] = permute_selectors [element_selector_10];
    permute_selector.as_m64[1] = 0x08090A0B0C0D0E0Full;
  #endif
  result_union.as_m128i = vec_perm (from_union.as_m128i, from_union.as_m128i, permute_selector.as_m128i);
  #ifdef __LITTLE_ENDIAN__
    return result_union.as_m64[1];
  #elif __BIG_ENDIAN__
    return result_union.as_m64[0];
  #endif
}

#endif
