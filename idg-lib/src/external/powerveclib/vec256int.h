/******************************************************************************/
/*                                                                            */
/* Licensed Materials - Property of IBM                                       */
/*                                                                            */
/* IBM Power Vector Intrinisic Functions version 1.0.6                        */
/*                                                                            */
/* Copyright IBM Corp. 2015,2016                                              */
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

#ifndef _H_VEC256INT
#define _H_VEC256INT

#include <altivec.h>
#include "veclib_types.h"

/******************************************************** Set *********************************************************/

/* Splat 32-bit int into 8 32-bit ints */
VECLIB_INLINE __m256i vec_splat8sw (int scalar)
{
  __m256i result;
  result.m128i_0 = vec_splat4sw (scalar);
  result.m128i_1 = vec_splat4sw (scalar);
  return result;
}

/* Set 8 32-bit ints */
VECLIB_INLINE __m256i vec_set8sw (int i7, int i6, int i5, int i4, int i3, int i2, int i1, int i0)
{
  __m256i_union result_union;
  #ifdef __LITTLE_ENDIAN__
    result_union.as_int[0] = i0;
    result_union.as_int[1] = i1;
    result_union.as_int[2] = i2;
    result_union.as_int[3] = i3;
    result_union.as_int[4] = i4;
    result_union.as_int[5] = i5;
    result_union.as_int[6] = i6;
    result_union.as_int[7] = i7;
  #elif __BIG_ENDIAN__
    result_union.as_int[0] = i7;
    result_union.as_int[1] = i6;
    result_union.as_int[2] = i5;
    result_union.as_int[3] = i4;
    result_union.as_int[4] = i3;
    result_union.as_int[5] = i2;
    result_union.as_int[6] = i1;
    result_union.as_int[7] = i0;
  #endif
  return result_union.as_m256i;
}

/******************************************************* Load *********************************************************/

/* Load 256 bits of integer data, word aligned */
VECLIB_INLINE __m256i vec_load8sw (__m256i const* from) {
  __m256_all_union result_union;
  result_union.as_m256 = vec_load8sp( (float*)from);
  return result_union.as_m256i;
}

/****************************************************** Store *********************************************************/

/* Store 256-bits of integer data, aligned */
VECLIB_INLINE void vec_store2q (__m256i* to, __m256i from)
{
  vec_st (from.m128i_0, 0, (vector unsigned char *) to);
  vec_st (from.m128i_1, 16, (vector unsigned char *) to);
}

/********************************************* Convert Floating Point to Integer **************************************/

/* Convert 8 32-bit floats to 8 32-bit ints with truncation */
VECLIB_INLINE __m256i vec_converttruncating8spto8sw (__m256 from)
{
  __m256i result;
  result.m128i_0 = (__m128i) vec_cts ((vector float) from.m128_0, 0);
  result.m128i_1 = (__m128i) vec_cts ((vector float) from.m128_1, 0);
  return result;
}

/* Convert 4 32-bit floats to 4 32-bit ints with rounding using current mode */
VECLIB_INLINE __m256i vec_convert8spto8sw (__m256 from)
{
  __m256_union from_union;
  from_union.as_m256 = from;

  /* Round to nearest with ties away from zero */
  vector float rounded_to_nearest_ties_away_0 = vec_round ((vector float) from.m128_0);
  vector float rounded_to_nearest_ties_away_1 = vec_round ((vector float) from.m128_1);

  /* vec_cts truncates towards zero */
  __m256i_union result;
  result.as_vector_signed_int[0] = vec_cts (rounded_to_nearest_ties_away_0, 0);
  result.as_vector_signed_int[1] = vec_cts (rounded_to_nearest_ties_away_1, 0);
  return result.as_m256i;
}

/***************************************************** Arithmetic *****************************************************/

/* Add 8 32-bit ints */
VECLIB_INLINE __m256i vec_add8sw (__m256i left, __m256i right)
{
  __m256i_union left_union; left_union.as_m256i = left;
  __m256i_union right_union; right_union.as_m256i = right;
  __m256i result;
  result.m128i_0 = (__m128i) vec_add ((vector int) left_union.as_m128i[0], (vector int) right_union.as_m128i[0]);
  result.m128i_1 = (__m128i) vec_add ((vector int) left_union.as_m128i[1], (vector int) right_union.as_m128i[1]);
  return result;
}

/******************************************************** Boolean *****************************************************/

/* Bitwise 256-bit and */
VECLIB_INLINE __m256i vec_bitand2q (__m256i left, __m256i right)
{
  __m256i_union left_union; left_union.as_m256i = left;
  __m256i_union right_union; right_union.as_m256i = right;
  __m256i result;
  result.m128i_0 = (__m128i) vec_and ((vector int) left_union.as_m128i[0], (vector int) right_union.as_m128i[0]);
  result.m128i_1 = (__m128i) vec_and ((vector int) left_union.as_m128i[1], (vector int) right_union.as_m128i[1]);
  return result;
}

#endif
