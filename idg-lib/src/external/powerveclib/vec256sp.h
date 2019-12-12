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

#ifndef _H_VEC256SP
#define _H_VEC256SP

#include <altivec.h>
#include "veclib_types.h"

/******************************************************** Load ********************************************************/

/* Load 8 32-bit floats, unaligned */
VECLIB_INLINE __m256 vec_loadu8sp (float const* from)
{
  __m256_union result_union;
  #ifdef __LITTLE_ENDIAN__
    #ifdef __ibmxl__
      result_union.as_m128[0] = vec_xl (0, (float *) from);
      result_union.as_m128[1] = vec_xl (16, (float *) from);
    #elif (defined __GNUC__) && (__GCC_VERSION__ >= 492)
      result_union.as_m128[0] = vec_xl (0, (float *) from);
      result_union.as_m128[1] = vec_xl (16, (float *) from);
    #elif (defined __GNUC__) && (__GCC_VERSION__ < 492)
      /* Prepare for later generate select control mask vector */
      vector unsigned int all_one = vec_splat_u32(0xFFFFFFFFu);
      vector unsigned int all_zero = vec_splat_u32( 0 );
      vector unsigned char permute_selector = vec_lvsr (0, from);
      vector unsigned int select_vector = vec_perm (all_one, all_zero, permute_selector);
      permute_selector = vec_andc ((vector unsigned char) all_one, permute_selector);
      /* load from[31:0] */
      __m128 temp_ld0 = vec_ld (0, from);
      __m128 temp_ld16 = vec_ld (16, from);
      temp_ld0 = vec_perm (temp_ld0, temp_ld0, permute_selector);
      temp_ld16 = vec_perm (temp_ld16, temp_ld16, permute_selector);
      result_union.as_m128[0] = vec_sel (temp_ld0, temp_ld16, select_vector);
      /* Load from[63:32] */
      temp_ld0 = vec_ld (16, from);
      temp_ld16 = vec_ld (32, from);
      temp_ld0 = vec_perm (temp_ld0, temp_ld0, permute_selector);
      temp_ld16 = vec_perm (temp_ld16, temp_ld16, permute_selector);
      result_union.as_m128[1] = vec_sel (temp_ld0, temp_ld16, select_vector);
    #endif
  #elif __BIG_ENDIAN__
    __m256_union temp0;
    temp0.as_m128[0] = vec_ld (0, from);
    temp0.as_m128[1] = vec_ld (16, from);
    __m256_union temp1;
    temp1.as_m128[0] = vec_ld (16, from);
    temp1.as_m128[1] = vec_ld (32, from);
    result_union.as_m128[0] = vec_perm (temp0.as_m128[0], temp1.as_m128[0], vec_lvsl (0, (float *)from));
    result_union.as_m128[1] = vec_perm (temp0.as_m128[1], temp1.as_m128[1], vec_lvsl (16, (float *)from));
  #endif
  return result_union.as_m256;
}

/* Load 8 32-bit floats, word aligned */
VECLIB_INLINE __m256 vec_load8sp (float const* from)
{
  __m256_union result_union;
  result_union.as_m128[0] = vec_ld (0, (float *) from);
  result_union.as_m128[1] = vec_ld (16, (float *) from);
  return result_union.as_m256;
}

/* Load 1 32-bit float and splat into 8 32-bit floats */
VECLIB_INLINE __m256 vec_loadsplat8sp (float const* from)
{
  __m256_union result_union;
  float splat_val = *from;
  result_union.as_m128[0] = (__m128) vec_splats (splat_val);
  result_union.as_m128[1] = (__m128) vec_splats (splat_val);
  return result_union.as_m256;
}

/* Load or zero under mask high bit 8 32-bit floats */
VECLIB_INLINE __m256 vec_loadu8spunderbitmask (float const* from, __m256i mask) {
  __m256_all_union result;  result.as_m256 = vec_loadu8sp(from);

  __m256_all_union andableMask;
  andableMask.as_vector_bool_long_long[0] = (vector bool long long) { 0x0000000000000000ull, 0x0000000000000000ull};
  andableMask.as_vector_bool_long_long[1] = (vector bool long long) { 0x0000000000000000ull, 0x0000000000000000ull};

  #ifdef __LITTLE_ENDIAN__
    andableMask.as_m128i[0] = (__m128i)vec_cmplt( (vector signed int) mask.m128i_0 , (vector signed int) andableMask.as_m128i[0]);
    andableMask.as_m128i[1] = (__m128i)vec_cmplt( (vector signed int) mask.m128i_1 , (vector signed int) andableMask.as_m128i[1]);

    result.as_m256i.m128i_0 = vec_and( andableMask.as_m128i[0], result.as_m256i.m128i_0  );
    result.as_m256i.m128i_1 = vec_and( andableMask.as_m128i[1], result.as_m256i.m128i_1  );
  #elif __BIG_ENDIAN__
    andableMask.as_m128i[0] = (__m128i)vec_cmplt( (vector signed int) mask.m128i_0 , (vector signed int) andableMask.as_m128i[0]);
    andableMask.as_m128i[1] = (__m128i)vec_cmplt( (vector signed int) mask.m128i_1 , (vector signed int) andableMask.as_m128i[1]);
    static const vector unsigned char flipMask = { 0x0F, 0x0E, 0x0D, 0x0C, 0x0B, 0x0A, 0x09, 0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x00 };
    result.as_m256i.m128i_0 = vec_and( vec_perm( andableMask.as_m128i[1], andableMask.as_m128i[1] , flipMask ), result.as_m256i.m128i_0  );
    result.as_m256i.m128i_1 = vec_and( vec_perm( andableMask.as_m128i[0], andableMask.as_m128i[0] , flipMask ), result.as_m256i.m128i_1  );
  #endif

  return result.as_m256;
}

/* forward declaration */
VECLIB_INLINE void vec_store2q (__m256i* to, __m256i from);

/* Gather 8 32-bit floats */
VECLIB_INLINE __m256 vec_gather8sp (float const* base_addr, __m256i vindex)
{
  float dst[8];
  int idx[8];
  vec_store2q((__m256i *) idx, vindex);
  for (unsigned i = 0; i < 8; i++) {
      dst[i] = base_addr[idx[i]];
  }
  return vec_load8sp(dst);
}

/******************************************************** Set *********************************************************/

/* Set 8 32-bit float literals */
VECLIB_INLINE __m256 vec_set8sp (float f7, float f6, float f5, float f4, float f3, float f2, float f1, float f0)
{
  /* For little endian, f0 is element 0 on the right. For big endian, f0 is element 0 on the left. */
  __m256_union result_union;
  #ifdef __LITTLE_ENDIAN__
    result_union.as_float[0] = f0;
    result_union.as_float[1] = f1;
    result_union.as_float[2] = f2;
    result_union.as_float[3] = f3;
    result_union.as_float[4] = f4;
    result_union.as_float[5] = f5;
    result_union.as_float[6] = f6;
    result_union.as_float[7] = f7;
  #elif __BIG_ENDIAN__
    result_union.as_float[0] = f7;
    result_union.as_float[1] = f6;
    result_union.as_float[2] = f5;
    result_union.as_float[3] = f4;
    result_union.as_float[4] = f3;
    result_union.as_float[5] = f2;
    result_union.as_float[6] = f1;
    result_union.as_float[7] = f0;
  #endif
  return result_union.as_m256;
}


/********************************************************* Splat ******************************************************/

/* Splat 32-bit float into 8 32-bit floats */
VECLIB_INLINE __m256 vec_splat8sp (float scalar)
{
  __m256 result;
  result.m128_0 = vec_splats (scalar);
  result.m128_1 = vec_splats (scalar);
  return result;
}


/********************************************************* Store ******************************************************/

/* Store 8 32-bit floats, unaligned */
VECLIB_INLINE void vec_storeu8sp (float* to, __m256 from)
{
  __m256_all_union from_union; from_union.as_m256 = from;
  #ifdef __LITTLE_ENDIAN__
    #ifdef __ibmxl__
      vec_xst (from_union.as_m128[0], 0, to);
      vec_xst (from_union.as_m128[1], 16, to);
    #elif (defined __GNUC__) && (__GCC_VERSION__ >= 492)
      vec_xst (from_union.as_m128[0], 0, to);
      vec_xst (from_union.as_m128[1], 16, to);
    #elif (defined __GNUC__) && (__GCC_VERSION__ < 492)
      /* Prepare for later generate select control mask vector */
      vector signed char all_one = vec_splat_s8( -1 );
      vector signed char all_zero = vec_splat_s8( 0 );
      /* Generate permute vector for the upper part of each m128d component */
      vector unsigned char permute_vector = vec_lvsl (0, (unsigned char *) to);
      permute_vector = vec_andc ((vector unsigned char) all_one, permute_vector);
      /* Generate selector vector for the lower part of each m128d component */
      vector unsigned char select_vector = vec_perm ((vector unsigned char) all_zero, (vector unsigned char) all_one, permute_vector);

      /* Load from.m128_0 */
      /* Perform a 16-byte load of the original data starting at BoundAlign (to + 0) and BoundAlign (to + 16) */
      vector unsigned char low0 = vec_ld (0, (unsigned char *) to);
      vector unsigned char high0 = vec_ld (16, (unsigned char *) to);
      /* Perform permute, the result will look like:
         original data ... from_union.as_vector_unsigned_char[0] ... original data */
      vector unsigned char temp_low0 = vec_perm (from_union.as_vector_unsigned_char[0], from_union.as_vector_unsigned_char[0], permute_vector);
      low0 = vec_sel (low0, temp_low0, select_vector);
      high0 = vec_perm (from_union.as_vector_unsigned_char[0], high0, permute_vector);
      /* Store the aligned result for from_union.as_m128[0] */
      vec_st (low0, 0, (unsigned char *) to);
      vec_st (high0, 16, (unsigned char *) to);

      /* Load from.m128_1 */
      /* Perform a 16-byte load of the original data starting at BoundAlign (to + 16) and BoundAlign (to + 32) */
      vector unsigned char low1 = vec_ld (16, (unsigned char *) to);
      vector unsigned char high1 = vec_ld (32, (unsigned char *) to);
      /* Perform permute, the result will look like:
         original data ... from_union.as_vector_unsigned_char[1] ... original data */
      vector unsigned char temp_low1 = vec_perm (from_union.as_vector_unsigned_char[1], from_union.as_vector_unsigned_char[1], permute_vector);
      low1 = vec_sel (low1, temp_low1, select_vector);
      high1 = vec_perm (from_union.as_vector_unsigned_char[1], high1, permute_vector);
      /* Store the aligned result for from_union.as_m128[0] */
      vec_st (low1, 16, (unsigned char *) to);
      vec_st (high1, 32, (unsigned char *) to);
    #endif
  #elif __BIG_ENDIAN__
    /* Prepare for later generate control mask vector */
    vector signed char all_one = vec_splat_s8( -1 );
    vector signed char all_zero = vec_splat_s8( 0 );
    /* Generate permute vector for the upper part of each m128d component */
    vector unsigned char permute_vector = vec_lvsr (0, (unsigned char *) to);
    /* Generate selector vector for the lower part of each m128d component */
    vector unsigned char select_vector = vec_perm ((vector unsigned char) all_zero, (vector unsigned char) all_one, permute_vector);

    /* Load from.m128_0 */
    /* Perform a 16-byte load of the original data starting at BoundAlign (to + 0) and BoundAlign (to + 16) */
    vector unsigned char low0 = vec_ld (0, (unsigned char *) to);
    vector unsigned char high0 = vec_ld (16, (unsigned char *) to);
    /* Perform permute, the result will look like:
       original data ... from_union.as_vector_unsigned_char[0] ... original data */
    vector unsigned char temp_low0 = vec_perm (low0, from_union.as_vector_unsigned_char[0], permute_vector);
    low0 = vec_sel (low0, temp_low0, select_vector);
    high0 = vec_perm (from_union.as_vector_unsigned_char[0], high0, permute_vector);
    /* Store the aligned result for from_union.as_m128[0] */
    vec_st (low0, 0, (unsigned char *) to);
    vec_st (high0, 16, (unsigned char *) to);

    /* Load from.m128_1 */
    /* Perform a 16-byte load of the original data starting at BoundAlign (to + 16) and BoundAlign (to + 32) */
    vector unsigned char low1 = vec_ld (16, (unsigned char *) to);
    vector unsigned char high1 = vec_ld (32, (unsigned char *) to);
    /* Perform permute, the result will look like:
       original data ... from_union.as_vector_unsigned_char[1] ... original data */
    vector unsigned char temp_low1 = vec_perm (low1, from_union.as_vector_unsigned_char[1], permute_vector);
    low1 = vec_sel (low1, temp_low1, select_vector);
    high1 = vec_perm (from_union.as_vector_unsigned_char[1], high1, permute_vector);
    /* Store the aligned result for from_union.as_m128[0] */
    vec_st (low1, 16, (unsigned char *) to);
    vec_st (high1, 32, (unsigned char *) to);
  #endif
}

/* Store 8 32-bit floats, unaligned - deprecated - use previous function */
VECLIB_INLINE void vec_store8sp (float* to, __m256 from)
{
  vec_storeu8sp (to, from);
  /* NOTE: aligned store also maps to vec_storeu8sp. After deprecated functions
           are phased out, vec_store8sp should be re-implemented */
  /* _mm256_store_ps: Store 8 32-bit floats, aligned 32 */
}


/******************************************************* Insert *******************************************************/

/* Insert 32-bit float quad into lower or upper half */
VECLIB_INLINE __m256 vec_insert4spinto8sp (__m256 into, __m128 from, intlit2 element_number)
{
  __m256_union into_union;
  into_union.as_m128[0] = into.m128_0;
  into_union.as_m128[1] = into.m128_1;
  /* For little endian, element 0 is on the right. For big endian, element 0 is on the left. */
  __m256_union result;
  if ((element_number & 1) == 0)
  {
    #ifdef __LITTLE_ENDIAN__
      result.as_m128[0] = from;
      result.as_m128[1] = into_union.as_m128[1];
    #elif __BIG_ENDIAN__
      result.as_m128[0] = into_union.as_m128[0];
      result.as_m128[1] = from;
    #endif
  }
  if ((element_number & 1) == 1)
  {
    #ifdef __LITTLE_ENDIAN__
      result.as_m128[0] = into_union.as_m128[0];
      result.as_m128[1] = from;
    #elif __BIG_ENDIAN__
      result.as_m128[0] = from;
      result.as_m128[1] = into_union.as_m128[1];
  #endif
  }
  return result.as_m256;
}

/********************************************** Convert Integer to Floating Point *************************************/

/* Convert 8 32-bit ints to 8 32-bit floats */
VECLIB_INLINE __m256 vec_convert8swto8sp (__m256i from)
{
  __m256 result;
  result.m128_0 = (__m128) vec_ctf ((vector signed int) from.m128i_0, 0);
  result.m128_1 = (__m128) vec_ctf ((vector signed int) from.m128i_1, 0);
  return result;
}

/****************************************************** Arithmetic ****************************************************/

/* Add 8 32-bit floats + 8 32-bit floats */
VECLIB_INLINE __m256 vec_add8sp (__m256 left, __m256 right)
{
  __m256_union left_union; left_union.as_m256 = left;
  __m256_union right_union; right_union.as_m256 = right;
  __m256 result;
  result.m128_0 = (__m128) vec_add ((vector float) left_union.as_m128[0], (vector float) right_union.as_m128[0]);
  result.m128_1 = (__m128) vec_add ((vector float) left_union.as_m128[1], (vector float) right_union.as_m128[1]);
  return result;
}

/* Subtract 8 32-bit floats - 8 32-bit floats */
VECLIB_INLINE __m256 vec_subtract8sp (__m256 left, __m256 right)
{
  __m256_union left_union; left_union.as_m256 = left;
  __m256_union right_union; right_union.as_m256 = right;
  __m256 result;
  result.m128_0 = (__m128) vec_sub ((vector float) left_union.as_m128[0], (vector float) right_union.as_m128[0]);
  result.m128_1 = (__m128) vec_sub ((vector float) left_union.as_m128[1], (vector float) right_union.as_m128[1]);
  return result;
}

/* Multiply 8 32-bit floats * 8 32-bit floats */
VECLIB_INLINE __m256 vec_multiply8sp (__m256 left, __m256 right)
{
  __m256_union left_union;   left_union.as_m256 = left;
  __m256_union right_union;  right_union.as_m256 = right;
  __m256 result;
  result.m128_0 = (__m128) vec_mul ((vector float) left_union.as_m128[0], (vector float) right_union.as_m128[0]);
  result.m128_1 = (__m128) vec_mul ((vector float) left_union.as_m128[1], (vector float) right_union.as_m128[1]);
  return result;
}

/* Divide 8 32-bit floats / 8 32-bit floats */
VECLIB_INLINE __m256 vec_divide8sp (__m256 left, __m256 right)
{
  __m256_union left_union;   left_union.as_m256 = left;
  __m256_union right_union;  right_union.as_m256 = right;
  __m256 result;
  result.m128_0 = (__m128) vec_div ((vector float) left_union.as_m128[0], (vector float) right_union.as_m128[0]);
  result.m128_1 = (__m128) vec_div ((vector float) left_union.as_m128[1], (vector float) right_union.as_m128[1]);
  return result;
}

/* Horizontally add 4+4 adjacent pairs of 32-bit floats to 8 32-bit floats */
VECLIB_INLINE __m256 vec_partialhorizontaladd32sp (__m256 left, __m256 right)
{
  __m256_union left_union; left_union.as_m256 = left;
  __m256_union right_union; right_union.as_m256 = right;
  __m256_union result_union;
  #ifdef __LITTLE_ENDIAN__
    static vector unsigned char addend_1_permute_mask = (vector unsigned char)
      { 0x00,0x01,0x02,0x03, 0x08,0x09,0x0A,0x0B, 0x10,0x11,0x12,0x13, 0x18,0x19,0x1A,0x1B };
    static vector unsigned char addend_2_permute_mask = (vector unsigned char)
      { 0x04,0x05,0x06,0x07, 0x0C,0x0D,0x0E,0x0F, 0x14,0x15,0x16,0x17, 0x1C,0x1D,0x1E,0x1F };
  #elif __BIG_ENDIAN__
    static vector unsigned char addend_1_permute_mask = (vector unsigned char)
      { 0x14,0x15,0x16,0x17, 0x1C,0x1D,0x1E,0x1F, 0x04,0x05,0x06,0x07, 0x0C,0x0D,0x0E,0x0F };
    static vector unsigned char addend_2_permute_mask = (vector unsigned char)
      { 0x10,0x11,0x12,0x13, 0x18,0x19,0x1A,0x1B, 0x00,0x01,0x02,0x03, 0x08,0x09,0x0A,0x0B };
  #endif
  vector float addend_1 = vec_perm (left_union.as_m128[0], right_union.as_m128[0], addend_1_permute_mask);
  vector float addend_2 = vec_perm (left_union.as_m128[0], right_union.as_m128[0], addend_2_permute_mask);
  vector float addend_3 = vec_perm (left_union.as_m128[1], right_union.as_m128[1], addend_1_permute_mask);
  vector float addend_4 = vec_perm (left_union.as_m128[1], right_union.as_m128[1], addend_2_permute_mask);
  result_union.as_m128[0] = vec_add (addend_1, addend_2);
  result_union.as_m128[1] = vec_add (addend_3, addend_4);
  return result_union.as_m256;
}

/* Square roots of 8 32-bit floats */
VECLIB_INLINE __m256 vec_Squareroot8sp (__m256 v) {
  __m256 result = v;
  result.m128_0 = vec_sqrt( v.m128_0);
  result.m128_1 = vec_sqrt( v.m128_1);
  return result;
}

/******************************************************** Permute *****************************************************/

/* Permute 8 32-bit floats */
VECLIB_INLINE __m256 vec_permute8sp (__m256 left, __m256 right, const intlit8 selectors)
{
  __m256 result;
  unsigned long element_selector_10 =  selectors       & 0x03;
  unsigned long element_selector_32 = (selectors >> 2) & 0x03;
  unsigned long element_selector_54 = (selectors >> 4) & 0x03;
  unsigned long element_selector_76 = (selectors >> 6) & 0x03;
  #ifdef __LITTLE_ENDIAN__
    const static unsigned int permute_selectors_from_left_operand  [4] = { 0x03020100u, 0x07060504u, 0x0B0A0908u, 0x0F0E0D0Cu };
    const static unsigned int permute_selectors_from_right_operand [4] = { 0x13121110u, 0x17161514u, 0x1B1A1918u, 0x1F1E1D1Cu };
  #elif __BIG_ENDIAN__
    const static unsigned int permute_selectors_from_left_operand  [4] = { 0x0C0D0E0Fu, 0x08090A0Bu, 0x04050607u, 0x00010203u };
    const static unsigned int permute_selectors_from_right_operand [4] = { 0x1C1D1E1Fu, 0x18191A1Bu, 0x14151617u, 0x10111213u };
  #endif
  __m128i_union permute_selectors;
  #ifdef __LITTLE_ENDIAN__
    permute_selectors.as_int[0] = permute_selectors_from_left_operand [element_selector_10];
    permute_selectors.as_int[1] = permute_selectors_from_left_operand [element_selector_32];
    permute_selectors.as_int[2] = permute_selectors_from_right_operand[element_selector_54];
    permute_selectors.as_int[3] = permute_selectors_from_right_operand[element_selector_76];
  #elif __BIG_ENDIAN__
    permute_selectors.as_int[3] = permute_selectors_from_left_operand [element_selector_10];
    permute_selectors.as_int[2] = permute_selectors_from_left_operand [element_selector_32];
    permute_selectors.as_int[1] = permute_selectors_from_right_operand[element_selector_54];
    permute_selectors.as_int[0] = permute_selectors_from_right_operand[element_selector_76];
  #endif
  result.m128_0 = (__m128) vec_perm ((vector unsigned char) left.m128_0, (vector unsigned char) right.m128_0,
                                  permute_selectors.as_vector_unsigned_char);
  result.m128_1 = (__m128) vec_perm ((vector unsigned char) left.m128_1, (vector unsigned char) right.m128_1,
                                  permute_selectors.as_vector_unsigned_char);
  return result;
}

/* Select 8+8 32-bit floats under bit mask */
VECLIB_INLINE __m256 vec_selectbybitmask8sp (__m256 left, __m256 right, const intlit8 element_selectors)
{
  #ifdef __LITTLE_ENDIAN__
    /* In big endian mode the element selectors bit mask is in reverse order */
    unsigned long lower_element_selectors =  element_selectors       & 0xF;
    unsigned long upper_element_selectors = (element_selectors >> 4) & 0xF;
  #elif __BIG_ENDIAN__
    /* In big endian mode the element selectors bit mask is in reverse order */
    unsigned long lower_element_selectors = (element_selectors >> 4) & 0xF;
    unsigned long upper_element_selectors = element_selectors       & 0xF;
  #endif
  static const vector bool int selectors [16] = {
    /* to select left if bit is 0 else right if it is 1 */
    #ifdef __LITTLE_ENDIAN__
      /* leftmost bit for leftmost element, rightmost bit for rightmost element */
      /* little endian means the first element below will be rightmost in a VR */
      { 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u },  /* 0000 */
      { 0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u },  /* 0001 */
      { 0x00000000u, 0xFFFFFFFFu, 0x00000000u, 0x00000000u },  /* 0010 */
      { 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u, 0x00000000u },  /* 0011 */
      { 0x00000000u, 0x00000000u, 0xFFFFFFFFu, 0x00000000u },  /* 0100 */
      { 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu, 0x00000000u },  /* 0101 */
      { 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u },  /* 0110 */
      { 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u },  /* 0111 */
      { 0x00000000u, 0x00000000u, 0x00000000u, 0xFFFFFFFFu },  /* 1000 */
      { 0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0xFFFFFFFFu },  /* 1001 */
      { 0x00000000u, 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu },  /* 1010 */
      { 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu },  /* 1011 */
      { 0x00000000u, 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu },  /* 1100 */
      { 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu },  /* 1101 */
      { 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu },  /* 1110 */
      { 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu },  /* 1111 */
    #elif __BIG_ENDIAN__
      /* leftmost bit for rightmost element, rightmost bit for leftmost element */
      /* big endian means the first element below will be leftmost in a VR */
      { 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u },  /* 0000 */
      { 0x00000000u, 0x00000000u, 0x00000000u, 0xFFFFFFFFu },  /* 0001 */
      { 0x00000000u, 0x00000000u, 0xFFFFFFFFu, 0x00000000u },  /* 0010 */
      { 0x00000000u, 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu },  /* 0011 */
      { 0x00000000u, 0xFFFFFFFFu, 0x00000000u, 0x00000000u },  /* 0100 */
      { 0x00000000u, 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu },  /* 0101 */
      { 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u },  /* 0110 */
      { 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu },  /* 0111 */
      { 0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u },  /* 1000 */
      { 0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0xFFFFFFFFu },  /* 1001 */
      { 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu, 0x00000000u },  /* 1010 */
      { 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu },  /* 1011 */
      { 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u, 0x00000000u },  /* 1100 */
      { 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu },  /* 1101 */
      { 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u },  /* 1110 */
      { 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu },  /* 1111 */
    #endif
  };
  __m256 result;
  result.m128_0 = (__m128) vec_sel ((vector float) left.m128_0, (vector float) right.m128_0, selectors[lower_element_selectors]);
  result.m128_1 = (__m128) vec_sel ((vector float) left.m128_1, (vector float) right.m128_1, selectors[upper_element_selectors]);
  return result;
}

/* Select 8+8 32-bit floats under bit mask - deprecated - use previous function */
VECLIB_INLINE __m256 vec_selectbybitmask88sp (__m256 left, __m256 right, const intlit8 element_selectors)
{
  return vec_selectbybitmask8sp (left, right, element_selectors);
}

/* Blend 8+8 32-bit floats under vector mask */
VECLIB_INLINE __m256 vec_selectbyvectormask8sp (__m256 left, __m256 right, __m256 mask)
{
  __m256_union left_union;  left_union.as_m256  = left;
  __m256_union right_union;  right_union.as_m256 = right;
  __m256_all_union mask_union;  mask_union.as_m256 = mask;
  /* Expand upper bit of each mask element to full element width */
  vector bool int mask_0 = vec_cmplt (mask_union.as_vector_signed_int[0], vec_splats (0));
  vector bool int mask_1 = vec_cmplt (mask_union.as_vector_signed_int[1], vec_splats (0));
  __m256 result;
  result.m128_0 = (__m128) vec_sel ((vector float) left.m128_0, (vector float) right.m128_0, mask_0);
  result.m128_1 = (__m128) vec_sel ((vector float) left.m128_1, (vector float) right.m128_1, mask_1);
  return result;
}

/* Blend 8+8 32-bit floats under vector mask - deprecated - use previous function */
VECLIB_INLINE __m256 vec_selectbyvectormask88sp (__m256 left, __m256 right, __m256 mask)
{
  return vec_selectbyvectormask8sp (left, right, mask);
}

/* Permute 2x2 128-bit (4 32-bit floats) */
VECLIB_INLINE __m256 vec_permute2q8sp (__m256 left, __m256 right, intlit8 selectors)
{
  __m256_union left_union; left_union.as_m256 = left;
  __m256_union right_union; right_union.as_m256 = right;
  __m256_union result_union;
  /* Permute Selectors:  2 4-bit selectors
   selectors[1:0]
     0 => left  source lower half
     1 => left  source upper half
     2 => right source lower half
     3 => right source upper half
   If selectors[3] = 1
     => 128-bits zeros */
  unsigned int low_selector = selectors & 0x03;
  unsigned int low_ctrl_zero = (selectors >> 3) & 0x01;
  unsigned int high_selector = (selectors >> 4) & 0x03;
  unsigned int high_ctrl_zero = (selectors >> 7) & 0x01;
  #ifdef __LITTLE_ENDIAN__
    if (low_ctrl_zero == 1)
    {
      result_union.as_m128[0] = (__m128) vec_splats(0.);
    }
    else if (low_selector == 0)
    {
      result_union.as_m128[0] = left_union.as_m128[0];
    }
    else if (low_selector == 1)
    {
      result_union.as_m128[0] = left_union.as_m128[1];
    }
    else if (low_selector == 2)
    {
      result_union.as_m128[0] = right_union.as_m128[0];
    }
    else if (low_selector == 3)
    {
      result_union.as_m128[0] = right_union.as_m128[1];
    }

    if (high_ctrl_zero == 1)
    {
      result_union.as_m128[1] = (__m128) vec_splats(0.);
    }
    else if (high_selector == 0)
    {
      result_union.as_m128[1] = left_union.as_m128[0];
    }
    else if (high_selector == 1)
    {
      result_union.as_m128[1] = left_union.as_m128[1];
    }
    else if (high_selector == 2)
    {
      result_union.as_m128[1] = right_union.as_m128[0];
    }
    else if (high_selector == 3)
    {
      result_union.as_m128[1] = right_union.as_m128[1];
    }
  #elif __BIG_ENDIAN__
    if (low_ctrl_zero == 1)
    {
      result_union.as_m128[1] = (__m128) vec_splats(0.);
    }
    else if (low_selector == 0)
    {
      result_union.as_m128[1] = left_union.as_m128[1];
    }
    else if (low_selector == 1)
    {
      result_union.as_m128[1] = left_union.as_m128[0];
    }
    else if (low_selector == 2)
    {
      result_union.as_m128[1] = right_union.as_m128[1];
    }
    else if (low_selector == 3)
    {
      result_union.as_m128[1] = right_union.as_m128[0];
    }

    if (high_ctrl_zero == 1)
    {
      result_union.as_m128[0] = (__m128) vec_splats(0.);
    }
    else if (high_selector == 0)
    {
      result_union.as_m128[0] = left_union.as_m128[1];
    }
    else if (high_selector == 1)
    {
      result_union.as_m128[0] = left_union.as_m128[0];
    }
    else if (high_selector == 2)
    {
      result_union.as_m128[0] = right_union.as_m128[1];
    }
    else if (high_selector == 3)
    {
      result_union.as_m128[0] = right_union.as_m128[0];
    }
  #endif
  return result_union.as_m256;
}

/******************************************************* Compare ******************************************************/

/* Compare under condition */

  /* Compare Conditions */
  #define _CMP_EQ_OQ    0
  #define _CMP_LT_OS    1
  #define _CMP_LE_OS    2
  #define _CMP_UNORD_Q  3
  #define _CMP_NEQ_UQ   4
  #define _CMP_NLT_US   5
  #define _CMP_NLE_US   6
  #define _CMP_ORD_Q    7
  #define _CMP_EQ_UQ    8
  #define _CMP_NGE_US   9
  #define _CMP_NGT_US   10
  #define _CMP_FALSE_OQ 11
  #define _CMP_NEQ_OQ   12
  #define _CMP_GE_OS    13
  #define _CMP_GT_OS    14
  #define _CMP_TRUE_UQ  15
  #define _CMP_EQ_OS    16
  #define _CMP_LT_OQ    17
  #define _CMP_LE_OQ    18
  #define _CMP_UNORD_S  19
  #define _CMP_NEQ_US   20
  #define _CMP_NLT_UQ   21
  #define _CMP_NLE_UQ   22
  #define _CMP_ORD_S    23
  #define _CMP_EQ_US    24
  #define _CMP_NGE_UQ   25
  #define _CMP_NGT_UQ   26
  #define _CMP_FALSE_OS 27
  #define _CMP_NEQ_OS   28
  #define _CMP_GE_OQ    29
  #define _CMP_GT_OQ    30
  #define _CMP_TRUE_US  31

/* Compare 8 32-bit floats for condition to mask */
VECLIB_INLINE __m256 vec_compare8sp (__m256 left, __m256 right, const intlit5 condition)
/* On PowerPC SIMD subnormals are always treated as zero. */
{
  __m256_all_union left_union;   left_union.as_m256 = left;
  __m256_all_union right_union;  right_union.as_m256 = right;
  __m256_all_union result_union;
  vector bool int temp0;
  vector bool int temp1;
  if ((condition & 7) == _CMP_EQ_OQ)  /*000*/
  {
    result_union.as_vector_bool_int[0] = vec_cmpeq (left_union.as_vector_float[0], right_union.as_vector_float[0]);
    result_union.as_vector_bool_int[1] = vec_cmpeq (left_union.as_vector_float[1], right_union.as_vector_float[1]);
  }
  if ((condition & 7) == _CMP_LT_OS)  /*001*/
  {
    result_union.as_vector_bool_int[0] = vec_cmplt (left_union.as_vector_float[0], right_union.as_vector_float[0]);
    result_union.as_vector_bool_int[1] = vec_cmplt (left_union.as_vector_float[1], right_union.as_vector_float[1]);
  }
  if ((condition & 7) == _CMP_LE_OS)  /*010*/
  {
    result_union.as_vector_bool_int[0] = vec_cmple (left_union.as_vector_float[0], right_union.as_vector_float[0]);
    result_union.as_vector_bool_int[1] = vec_cmple (left_union.as_vector_float[1], right_union.as_vector_float[1]);
  }
  if ((condition & 7) == _CMP_UNORD_Q)  /*011*/
  {
    temp0 = vec_or (vec_cmplt (left_union.as_vector_float[0], right_union.as_vector_float[0]),
                    vec_cmpge (left_union.as_vector_float[0], right_union.as_vector_float[0]));
    temp1 = vec_or (vec_cmplt (left_union.as_vector_float[1], right_union.as_vector_float[1]),
                    vec_cmpge (left_union.as_vector_float[1], right_union.as_vector_float[1]));
    result_union.as_vector_bool_int[0] = vec_nor (temp0, temp0);
    result_union.as_vector_bool_int[1] = vec_nor (temp1, temp1);
  }

  if ((condition & 7) == _CMP_NEQ_UQ)  /*100*/
  {
    temp0 = vec_cmpeq (left_union.as_vector_float[0], right_union.as_vector_float[0]);
    temp1 = vec_cmpeq (left_union.as_vector_float[1], right_union.as_vector_float[1]);
    result_union.as_vector_bool_int[0] = vec_nor (temp0, temp0);
    result_union.as_vector_bool_int[1] = vec_nor (temp1, temp1);
  }
  if ((condition & 7) == _CMP_NLT_US)  /*101*/
  {
    temp0 = vec_cmplt (left_union.as_vector_float[0], right_union.as_vector_float[0]);
    temp1 = vec_cmplt (left_union.as_vector_float[1], right_union.as_vector_float[1]);
    result_union.as_vector_bool_int[0] = vec_nor (temp0, temp0);
    result_union.as_vector_bool_int[1] = vec_nor (temp1, temp1);
  }
  if ((condition & 7) == _CMP_NLE_US)  /*110*/
  {
    temp0 = vec_cmple (left_union.as_vector_float[0], right_union.as_vector_float[0]);
    temp1 = vec_cmple (left_union.as_vector_float[1], right_union.as_vector_float[1]);
    result_union.as_vector_bool_int[0] = vec_nor (temp0, temp0);
    result_union.as_vector_bool_int[1] = vec_nor (temp1, temp1);
  }
  if ((condition & 7) == _CMP_ORD_Q)  /*111*/
  {
    result_union.as_vector_bool_int[0] = vec_or (vec_cmplt (left_union.as_vector_float[0], right_union.as_vector_float[0]),
                                                 vec_cmpge (left_union.as_vector_float[0], right_union.as_vector_float[0]));
    result_union.as_vector_bool_int[1] = vec_or (vec_cmplt (left_union.as_vector_float[1], right_union.as_vector_float[1]),
                                                 vec_cmpge (left_union.as_vector_float[1], right_union.as_vector_float[1]));
  }
  return result_union.as_m256;
}


/******************************************************** Boolean *****************************************************/

/* Bitwise 256-bit xor */
VECLIB_INLINE __m256 vec_bitwisexor8sp (__m256 left, __m256 right)
{
  __m256_union left_union; left_union.as_m256 = left;
  __m256_union right_union; right_union.as_m256 = right;
  __m256 result;
  result.m128_0 = (__m128) vec_xor ((vector float) left_union.as_m128[0], (vector float) right_union.as_m128[0]);
  result.m128_1 = (__m128) vec_xor ((vector float) left_union.as_m128[1], (vector float) right_union.as_m128[1]);
  return result;
}


/********************************************************* Cast *******************************************************/

/* Cast 4 32-bit floats to 4 32-bit floats and 4 undefined */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m256 vec_cast4spto4sp4u (__m128 from)
  {
    __m256_union result_union;
    #ifdef __LITTLE_ENDIAN__
      result_union.as_m128[0] = from;
      result_union.as_m128[1] = (vector float) {0.f, 0.f, 0.f, 0.f}; /* undefined */
    #elif __BIG_ENDIAN__
      result_union.as_m128[0] = (vector float) {0.f, 0.f, 0.f, 0.f};
      result_union.as_m128[1] = from; /* undefined */
    #endif
    return result_union.as_m256;
  }
#endif


/******************************************************** Unpack ******************************************************/

/* Unpack and interleave 4+4 32-bit floats from high half of each 128-bit half */
VECLIB_INLINE __m256 vec_unpackupper8spto8sp (__m256 to_even, __m256 to_odd)
{
  __m256_union to_even_union; to_even_union.as_m256 = to_even;
  __m256_union to_odd_union; to_odd_union.as_m256 = to_odd;
  __m256 result;
  static const vector unsigned char permute_selector = {
    #ifdef __LITTLE_ENDIAN__
      0x08, 0x09, 0x0A, 0x0B,  0x18, 0x19, 0x1A, 0x1B,  0x0C, 0x0D, 0x0E, 0x0F,  0x1C, 0x1D, 0x1E, 0x1F
    #elif __BIG_ENDIAN__
      0x10, 0x11, 0x12, 0x13,  0x00, 0x01, 0x02, 0x03,  0x14, 0x15, 0x16, 0x17,  0x04, 0x05, 0x06, 0x07
    #endif
  };
  result.m128_0 = vec_perm ((vector float) to_even_union.as_m128[0], (vector float) to_odd_union.as_m128[0], permute_selector);
  result.m128_1 = vec_perm ((vector float) to_even_union.as_m128[1], (vector float) to_odd_union.as_m128[1], permute_selector);
  return result;
}

/* Unpack and interleave 4+4 32-bit floats from high half of each 128-bit half - deprecated - use previous function */
VECLIB_INLINE __m256 vec_unpackupper88spto8sp (__m256 to_even, __m256 to_odd)
{
  return vec_unpackupper8spto8sp (to_even, to_odd);
}

/* Unpack and interleave 4+4 32-bit floats from low half of each 128-bit half */
VECLIB_INLINE __m256 vec_unpacklower8spto8sp (__m256 to_even, __m256 to_odd)
{
  __m256_union to_even_union; to_even_union.as_m256 = to_even;
  __m256_union to_odd_union; to_odd_union.as_m256 = to_odd;
  __m256 result;
  static const vector unsigned char permute_selector = {
    #ifdef __LITTLE_ENDIAN__
      0x00, 0x01, 0x02, 0x03,  0x10, 0x11, 0x12, 0x13,  0x04, 0x05, 0x06, 0x07,  0x14, 0x15, 0x16, 0x17
    #elif __BIG_ENDIAN__
      0x18, 0x19, 0x1A, 0x1B,  0x08, 0x09, 0x0A, 0x0B,  0x1C, 0x1D, 0x1E, 0x1F,  0x0C, 0x0D, 0x0E, 0x0F
    #endif
  };
  result.m128_0 = vec_perm ((vector float) to_even_union.as_m128[0], (vector float) to_odd_union.as_m128[0], permute_selector);
  result.m128_1 = vec_perm ((vector float) to_even_union.as_m128[1], (vector float) to_odd_union.as_m128[1], permute_selector);
  return result;
}

/* Unpack and interleave 4+4 32-bit floats from low half of each 128-bit half - deprecated - use previous function */
VECLIB_INLINE __m256 vec_unpacklower88spto8sp (__m256 to_even, __m256 to_odd)
{
  return vec_unpacklower8spto8sp (to_even, to_odd);
}

#endif
