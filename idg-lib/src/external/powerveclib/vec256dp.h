/******************************************************************************/
/*                                                                            */
/* Licensed Materials - Property of IBM                                       */
/*                                                                            */
/* IBM Power Vector Intrinisic Functions version 1.0.6                        */
/*                                                                            */
/* Copyright IBM Corp. 2014,2016                                              */
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

#ifndef _H_VEC256DP
#define _H_VEC256DP

#include <altivec.h>
#include "veclib_types.h"

/******************************************************** Load ********************************************************/

/* Load 4 64-bit doubles, unaligned */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m256d vec_loadu4dp (double const* from)
  {
    __m256_all_union result_union;
    #ifdef __LITTLE_ENDIAN__
      #ifdef __ibmxl__
        result_union.as_m128d[0] = vec_xl (0, (double *) from);
        result_union.as_m128d[1] = vec_xl (16, (double *) from);
      #elif (defined __GNUC__) && (__GCC_VERSION__ >= 492)
        result_union.as_m128d[0] = vec_xl (0, (double *) from);
        result_union.as_m128d[1] = vec_xl (16, (double *) from);
      #elif (defined __GNUC__) && (__GCC_VERSION__< 492)
        /* Prepare for later generate select control mask vector */
        vector unsigned char all_one = vec_splat_u8((signed char) 0xFF);
        vector unsigned char all_zero = vec_splat_u8( 0 );
        vector unsigned char permute_selector = vec_lvsr (0, from);
        vector unsigned char select_vector = vec_perm (all_one, all_zero, permute_selector);
        permute_selector = vec_andc ((vector unsigned char) all_one, permute_selector);
        /* Load from[31:0] */
        __m128_all_union temp0_union; __m128_all_union temp1_union;
        temp0_union.as_vector_unsigned_char = vec_ld (0, (const unsigned char *) from);
        temp1_union.as_vector_unsigned_char = vec_ld (16, (const unsigned char *) from);
        temp0_union.as_vector_unsigned_char = vec_perm (temp0_union.as_vector_unsigned_char,
                                                  temp0_union.as_vector_unsigned_char, permute_selector);
        temp1_union.as_vector_unsigned_char = vec_perm (temp1_union.as_vector_unsigned_char,
                                                  temp1_union.as_vector_unsigned_char, permute_selector);
        result_union.as_vector_unsigned_char[0] = vec_sel (temp0_union.as_vector_unsigned_char,
                                                                temp1_union.as_vector_unsigned_char, select_vector);
        /* Load from[63:32] */
        temp0_union.as_vector_unsigned_char = vec_ld (16, (const unsigned char *) from);
        temp1_union.as_vector_unsigned_char = vec_ld (32, (const unsigned char *) from);
        temp0_union.as_vector_unsigned_char = vec_perm (temp0_union.as_vector_unsigned_char,
                                                  temp0_union.as_vector_unsigned_char, permute_selector);
        temp1_union.as_vector_unsigned_char = vec_perm (temp1_union.as_vector_unsigned_char,
                                                  temp1_union.as_vector_unsigned_char, permute_selector);
        result_union.as_vector_unsigned_char[1] = vec_sel (temp0_union.as_vector_unsigned_char,
                                                                temp1_union.as_vector_unsigned_char, select_vector);
      #endif
    #elif __BIG_ENDIAN__
      __m256_all_union temp0;
      temp0.as_vector_unsigned_char[0] = vec_ld (0, (const unsigned char *) from);
      temp0.as_vector_unsigned_char[1] = vec_ld (16, (const unsigned char *) from);
      __m256_all_union temp1;
      temp1.as_vector_unsigned_char[0] = vec_ld (16, (const unsigned char *) from);
      temp1.as_vector_unsigned_char[1] = vec_ld (32, (const unsigned char *) from);
      result_union.as_vector_unsigned_char[0] = vec_perm (temp0.as_vector_unsigned_char[0], temp1.as_vector_unsigned_char[0], vec_lvsl (0, (unsigned char *) from));
      result_union.as_vector_unsigned_char[1] = vec_perm (temp0.as_vector_unsigned_char[1], temp1.as_vector_unsigned_char[1], vec_lvsl (16, (unsigned char *) from));
    #endif
    return result_union.as_m256d;
  }
#endif

/* Load 4 64-bit doubles, unaligned - deprecated - use previous function */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m256d vec_load4dp (double const* from)
  {
    return vec_loadu4dp (from);
  }
#endif

/******************************************************** Set *********************************************************/

/* Set 4 64-bit double literals */
#ifdef VECLIB_VSX
VECLIB_INLINE __m256d vec_set4dp (double d3, double d2, double d1, double d0)
{
  /*- For little endian, d0 is element 0 on the right. For big endian, d0 is element 0 on the left. */
  __m256d_union result_union;
  #ifdef __LITTLE_ENDIAN__
    result_union.as_double[0] = d0;
    result_union.as_double[1] = d1;
    result_union.as_double[2] = d2;
    result_union.as_double[3] = d3;
  #elif __BIG_ENDIAN__
    result_union.as_double[0] = d3;
    result_union.as_double[1] = d2;
    result_union.as_double[2] = d1;
    result_union.as_double[3] = d0;
  #endif
  return result_union.as_m256d;
}
#endif

/******************************************************** Splat *******************************************************/

/* Splat 64-bit double into 4 64-bit doubles */
#ifdef VECLIB_VSX
VECLIB_INLINE __m256d vec_splat4dp (double scalar)
{
  __m256d result;
  result.m128d_0 = vec_splats (scalar);
  result.m128d_1 = vec_splats (scalar);
  return result;
}
#endif

/******************************************************** Store *******************************************************/

/* Store 4 64-bit doubles, unaligned */
#ifdef VECLIB_VSX
  VECLIB_INLINE void vec_storeu4dp (double* to, __m256d from)
  {
    __m256_all_union from_union; from_union.as_m256d = from;
    #ifdef __LITTLE_ENDIAN__
      #ifdef __ibmxl__
        vec_xst ((vector double) from_union.as_m128d[0], 0, to);
        vec_xst ((vector double) from_union.as_m128d[1], 16, to);
      #elif (defined __GNUC__) && (__GCC_VERSION__ >= 492)
        vec_xst ((vector double) from_union.as_m128d[0], 0, to);
        vec_xst ((vector double) from_union.as_m128d[1], 16, to);
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
        /* Prepare for later generate select control mask vector */
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
#endif

/* Store 4 64-bit doubles, unaligned - deprecated - use previous function */
#ifdef VECLIB_VSX
  VECLIB_INLINE void vec_store4dp (double* to, __m256d from)
  {
    vec_storeu4dp (to, from);
  }
#endif

/******************************************************* Insert *******************************************************/

/* Insert 64-bit double pair into lower or upper half */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m256d vec_insert2dpinto4dp (__m256d into, __m128d from, intlit1 element_number)
  {
    __m256d_union into_union;  into_union.as_m256d = into;
    /* For little endian, element 0 is on the right. For big endian, element 0 is on the left. */
    __m256d_union result;
    if ((element_number & 1) == 0) {
    #ifdef __LITTLE_ENDIAN__
          result.as_m128d[0] = from;
          result.as_m128d[1] = into_union.as_m128d[1];
    #elif __BIG_ENDIAN__
          result.as_m128d[0] = into_union.as_m128d[0];
          result.as_m128d[1] = from;
    #endif
    }
    if ((element_number & 1) == 1) {
    #ifdef __LITTLE_ENDIAN__
          result.as_m128d[0] = into_union.as_m128d[0];
          result.as_m128d[1] = from;
    #elif __BIG_ENDIAN__
          result.as_m128d[0] = from;
          result.as_m128d[1] = into_union.as_m128d[1];
    #endif
    }
    return result.as_m256d;
  }
#endif


/********************************************** Convert Integer to Floating Point *************************************/

/* Convert low 4 32-bit ints to 4 64-bit doubles */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m256d vec_convert4swto4dp (__m128i from)
  {
    __m128i_union from_union;
    from_union.as_m128i = from;
    __m256d_union result;
    result.as_double[0] = (double) from_union.as_int[0];
    result.as_double[1] = (double) from_union.as_int[1];
    result.as_double[2] = (double) from_union.as_int[2];
    result.as_double[3] = (double) from_union.as_int[3];
    return result.as_m256d;
  }
#endif

/****************************************************** Arithmetic ****************************************************/

/* Add 4 64-bit doubles + 4 64-bit doubles */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m256d vec_add4dp (__m256d left, __m256d right)
  {
    __m256d_union left_union; left_union.as_m256d = left;
    __m256d_union right_union; right_union.as_m256d = right;
    __m256d result;
    result.m128d_0 = (__m128d) vec_add ((vector double) left_union.as_m128d[0], (vector double) right_union.as_m128d[0]);
    result.m128d_1 = (__m128d) vec_add ((vector double) left_union.as_m128d[1], (vector double) right_union.as_m128d[1]);
    return result;
  }
#endif

/* Subtract 4 64-bit doubles - 4 64-bit doubles */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m256d vec_subtract4dp (__m256d left, __m256d right)
  {
    __m256d_union left_union;  left_union.as_m256d = left;
    __m256d_union right_union;  right_union.as_m256d = right;
    __m256d result;
    result.m128d_0 = (__m128d) vec_sub ((vector double) left_union.as_m128d[0], (vector double) right_union.as_m128d[0]);
    result.m128d_1 = (__m128d) vec_sub ((vector double) left_union.as_m128d[1], (vector double) right_union.as_m128d[1]);
    return result;
  }
#endif

/* Multiply 4 64-bit doubles * 4 64-bit doubles */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m256d vec_multiply4dp (__m256d left, __m256d right)
  {
    __m256d_union left_union;  left_union.as_m256d = left;
    __m256d_union right_union;  right_union.as_m256d = right;
    __m256d result;
    result.m128d_0 = (__m128d) vec_mul ((vector double) left_union.as_m128d[0], (vector double) right_union.as_m128d[0]);
    result.m128d_1 = (__m128d) vec_mul ((vector double) left_union.as_m128d[1], (vector double) right_union.as_m128d[1]);
    return result;
}
#endif

/* Divide 4 64-bit doubles / 4 64-bit doubles */
#ifdef VECLIB_VSX
VECLIB_INLINE __m256d vec_divide4dp (__m256d left, __m256d right)
{
  __m256d_union left_union;  left_union.as_m256d = left;
  __m256d_union right_union;  right_union.as_m256d = right;
  __m256d result;
  result.m128d_0 = (__m128d) vec_div ((vector double) left_union.as_m128d[0], (vector double) right_union.as_m128d[0]);
  result.m128d_1 = (__m128d) vec_div ((vector double) left_union.as_m128d[1], (vector double) right_union.as_m128d[1]);
  return result;
}
#endif

/****************************************************** Boolean *******************************************************/

/* Bitwise 256-bit xor */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m256d vec_bitwisexor4dp (__m256d left, __m256d right)
  {
    __m256d_union left_union;  left_union.as_m256d = left;
    __m256d_union right_union;  right_union.as_m256d = right;
    __m256d result;
    result.m128d_0 = (__m128d) vec_xor ((vector double) left_union.as_m128d[0], (vector double) right_union.as_m128d[0]);
    result.m128d_1 = (__m128d) vec_xor ((vector double) left_union.as_m128d[1], (vector double) right_union.as_m128d[1]);
    return result;
  }
#endif


/******************************************************* Permute ******************************************************/

/* Permute 4+4 64-bit doubles */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m256d vec_permute4dp (__m256d left, __m256d right, const intlit4 selectors)
  {
    __m256d result;
    #ifdef __LITTLE_ENDIAN__
        unsigned int selector_0 = selectors & 0x3;
        unsigned int selector_1 = (selectors >> 2) & 0x3;
    #elif __BIG_ENDIAN__
        unsigned int selector_0 = (selectors >> 2) & 0x3;
        unsigned int selector_1 = selectors & 0x3;
    #endif
    static const vector unsigned char permute_selector[4] = {
        /* To select left element for 0 or right element for 1 */
        #ifdef __LITTLE_ENDIAN__
          { 0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07, 0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17 }, /* 00 */
          { 0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F, 0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17 }, /* 01 */
          { 0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07, 0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F }, /* 10 */
          { 0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F, 0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F }  /* 11 */
        #elif __BIG_ENDIAN__
          { 0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F, 0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F }, /* 00 */
          { 0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F, 0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07 }, /* 01 */
          { 0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17, 0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F }, /* 10 */
          { 0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17, 0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07 }  /* 00 */
        #endif
    };
    result.m128d_0 = (__m128d) vec_perm ((vector unsigned char) left.m128d_0, (vector unsigned char) right.m128d_0, permute_selector[selector_0]);
    result.m128d_1 = (__m128d) vec_perm ((vector unsigned char) left.m128d_1, (vector unsigned char) right.m128d_1, permute_selector[selector_1]);
    return result;
  }
#endif

/* Permute 4+4 64-bit doubles - deprecated - use previous function */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m256d vec_permute44dp (__m256d left, __m256d right, const intlit4 selectors)
  {
    return vec_permute4dp (left, right, selectors);
  }
#endif

/* Permute 2x2 128-bit (2 64-bit doubles) */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m256d vec_permute2q4dp (__m256d left, __m256d right, intlit8 selectors)
  {
    __m256d_union left_union; left_union.as_m256d = left;
    __m256d_union right_union; right_union.as_m256d = right;
    __m256d_union result_union;
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
          result_union.as_m128d[0] = (__m128d) vec_splats(0.);
        }
        else if (low_selector == 0)
        {
          result_union.as_m128d[0] = left_union.as_m128d[0];
        }
        else if (low_selector == 1)
        {
          result_union.as_m128d[0] = left_union.as_m128d[1];
        }
        else if (low_selector == 2)
        {
          result_union.as_m128d[0] = right_union.as_m128d[0];
        }
        else if (low_selector == 3)
        {
          result_union.as_m128d[0] = right_union.as_m128d[1];
        }

        if (high_ctrl_zero == 1)
        {
          result_union.as_m128d[1] = (__m128d) vec_splats(0.);
        }
        else if (high_selector == 0)
        {
          result_union.as_m128d[1] = left_union.as_m128d[0];
        }
        else if (high_selector == 1)
        {
          result_union.as_m128d[1] = left_union.as_m128d[1];
        }
        else if (high_selector == 2)
        {
          result_union.as_m128d[1] = right_union.as_m128d[0];
        }
        else if (high_selector == 3)
        {
          result_union.as_m128d[1] = right_union.as_m128d[1];
        }
    #elif __BIG_ENDIAN__
        if (low_ctrl_zero == 1)
        {
          result_union.as_m128d[1] = (__m128d) vec_splats(0.);
        }
        else if (low_selector == 0)
        {
          result_union.as_m128d[1] = left_union.as_m128d[1];
        }
        else if (low_selector == 1)
        {
          result_union.as_m128d[1] = left_union.as_m128d[0];
        }
        else if (low_selector == 2)
        {
          result_union.as_m128d[1] = right_union.as_m128d[1];
        }
        else if (low_selector == 3)
        {
          result_union.as_m128d[1] = right_union.as_m128d[0];
        }

        if (high_ctrl_zero == 1)
        {
          result_union.as_m128d[0] = (__m128d) vec_splats(0.);
        }
        else if (high_selector == 0)
        {
          result_union.as_m128d[0] = left_union.as_m128d[1];
        }
        else if (high_selector == 1)
        {
          result_union.as_m128d[0] = left_union.as_m128d[0];
        }
        else if (high_selector == 2)
        {
          result_union.as_m128d[0] = right_union.as_m128d[1];
        }
        else if (high_selector == 3)
        {
          result_union.as_m128d[0] = right_union.as_m128d[0];
        }
    #endif
    return result_union.as_m256d;
  }
#endif

/* Permute 2x2 128-bit (2 64-bit doubles) - deprecated - use previous function */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m256d vec_permute22q4dp (__m256d left, __m256d right, intlit8 selectors)
  {
    return vec_permute2q4dp (left, right, selectors);
  }
#endif

/* Select 4+4 64-bit doubles under bit mask */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m256d vec_selectbybitmask4dp (__m256d left, __m256d right, const intlit4 element_selectors)
  {
    #ifdef __LITTLE_ENDIAN__
      unsigned long lower_element_selectors =  element_selectors         & 3;
      unsigned long upper_element_selectors = (element_selectors >> 2) & 3;
    #elif __BIG_ENDIAN__
      unsigned long lower_element_selectors = (element_selectors >> 2) & 3;
      unsigned long upper_element_selectors =  element_selectors         & 3;
    #endif
    static const vector bool long long selectors [4] = {
      /* To select left if bit is 0 else right if it is 1 */
      #ifdef __LITTLE_ENDIAN__
        /* Leftmost bit for leftmost element, rightmost bit for rightmost element */
        /* Little endian means the first element below will be rightmost in a VR */
        { 0x0000000000000000ull, 0x0000000000000000ull },  /* 00 */
        { 0xFFFFFFFFFFFFFFFFull, 0x0000000000000000ull },  /* 01 */
        { 0x0000000000000000ull, 0xFFFFFFFFFFFFFFFFull },  /* 10 */
        { 0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull },  /* 11 */
      #elif __BIG_ENDIAN__
        /* Leftmost bit for rightmost element, rightmost bit for leftmost element */
        /* Big endian means the first element below will be leftmost in a VR */
        { 0x0000000000000000ull, 0x0000000000000000ull },  /* 00 */
        { 0x0000000000000000ull, 0xFFFFFFFFFFFFFFFFull },  /* 01 */
        { 0xFFFFFFFFFFFFFFFFull, 0x0000000000000000ull },  /* 10 */
        { 0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull },  /* 11 */
      #endif
    };
    __m256d result;
    result.m128d_0 = (__m128d) vec_sel ((vector double) left.m128d_0, (vector double) right.m128d_0, selectors[lower_element_selectors]);
    result.m128d_1 = (__m128d) vec_sel ((vector double) left.m128d_1, (vector double) right.m128d_1, selectors[upper_element_selectors]);
    return result;
  }
#endif

/* Select 4+4 64-bit doubles under bit mask - deprecated - use previous function */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m256d vec_selectbybitmask44dp (__m256d left, __m256d right, const intlit4 element_selectors)
  {
    return vec_selectbybitmask4dp (left, right, element_selectors);
  }
#endif

/* Select 4+4 64-bit doubles under vector mask */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m256d vec_selectbyvectormask4dp (__m256d left, __m256d right, __m256d mask)
  {
    __m256d_union left_union;  left_union.as_m256d  = left;
    __m256d_union right_union;  right_union.as_m256d = right;
    __m256_all_union mask_union;  mask_union.as_m256d = mask;
    /* Expand upper bit of each mask element to full element width */
    vector bool int int_mask_0 = vec_cmplt (mask_union.as_vector_signed_int[0], vec_splats (0));
    vector bool int int_mask_1 = vec_cmplt (mask_union.as_vector_signed_int[1], vec_splats (0));
    static const vector unsigned char permute_selector = {
      #ifdef __LITTLE_ENDIAN__
        0x04,0x05,0x06,0x07, 0x04,0x05,0x06,0x07, 0x0C,0x0D,0x0E,0x0F, 0x0C,0x0D,0x0E,0x0F
      #elif __BIG_ENDIAN__
        0x00,0x01,0x02,0x03, 0x00,0x01,0x02,0x03, 0x08,0x09,0x0A,0x0B, 0x08,0x09,0x0A,0x0B
      #endif
    };
    __m128_all_union mask_0;  mask_0.as_vector_bool_int = vec_perm (int_mask_0, int_mask_0, permute_selector);
    __m128_all_union mask_1;  mask_1.as_vector_bool_int = vec_perm (int_mask_1, int_mask_1, permute_selector);
    __m256d result;
    result.m128d_0 = (__m128d) vec_sel ((vector double) left.m128d_0, (vector double) right.m128d_0, mask_0.as_vector_bool_long_long);
    result.m128d_1 = (__m128d) vec_sel ((vector double) left.m128d_1, (vector double) right.m128d_1, mask_1.as_vector_bool_long_long);
    return result;
  }
#endif

/* Select 4+4 64-bit doubles under vector mask - deprecated - use previous function */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m256d vec_selectbyvectormask44dp (__m256d left, __m256d right, __m256d mask)
  {
    return vec_selectbyvectormask4dp (left, right, mask);
  }
#endif

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

/* Compare 4 64-bit doubles for condition to mask */
#ifdef VECLIB_VSX
VECLIB_INLINE __m256d vec_compare4dp (__m256d left, __m256d right, const intlit5 condition)
{
  __m256_all_union left_union;   left_union.as_m256d = left;
  __m256_all_union right_union;  right_union.as_m256d = right;
  __m256_all_union result_union;
  #ifdef __ibmxl__
    vector bool long long temp0;
    vector bool long long temp1;
  #else
    __m128_all_union temp0_union;
    __m128_all_union temp1_union;
    __m128_all_union tempx_union;
    __m128_all_union tempy_union;
  #endif
  if ((condition & 7) == _CMP_EQ_OQ)  /*000*/
  {
    result_union.as_vector_bool_long_long[0] = vec_cmpeq (left_union.as_vector_double[0], right_union.as_vector_double[0]);
    result_union.as_vector_bool_long_long[1] = vec_cmpeq (left_union.as_vector_double[1], right_union.as_vector_double[1]);
  }
  if ((condition & 7) == _CMP_LT_OS)  /*001*/
  {
    result_union.as_vector_bool_long_long[0] = vec_cmplt (left_union.as_vector_double[0], right_union.as_vector_double[0]);
    result_union.as_vector_bool_long_long[1] = vec_cmplt (left_union.as_vector_double[1], right_union.as_vector_double[1]);
  }
  if ((condition & 7) == _CMP_LE_OS)  /*010*/
  {
    result_union.as_vector_bool_long_long[0] = vec_cmple (left_union.as_vector_double[0], right_union.as_vector_double[0]);
    result_union.as_vector_bool_long_long[1] = vec_cmple (left_union.as_vector_double[1], right_union.as_vector_double[1]);
  }
  if ((condition & 7) == _CMP_UNORD_Q)  /*011*/
  {
    #ifdef __ibmxl__
      temp0 = vec_or (vec_cmplt (left_union.as_vector_double[0], right_union.as_vector_double[0]),
                          vec_cmpge (left_union.as_vector_double[0], right_union.as_vector_double[0]));
      temp1 = vec_or (vec_cmplt (left_union.as_vector_double[1], right_union.as_vector_double[1]),
                          vec_cmpge (left_union.as_vector_double[1], right_union.as_vector_double[1]));
      result_union.as_vector_bool_long_long[0] = vec_nor (temp0, temp0);
      result_union.as_vector_bool_long_long[1] = vec_nor (temp1, temp1);
    #else
      temp0_union.as_vector_bool_long_long = vec_cmplt (left_union.as_vector_double[0], right_union.as_vector_double[0]);
      temp1_union.as_vector_bool_long_long = vec_cmpge (left_union.as_vector_double[0], right_union.as_vector_double[0]);
      tempx_union.as_vector_double = vec_or (temp0_union.as_vector_double, temp1_union.as_vector_double);

      temp0_union.as_vector_bool_long_long = vec_cmplt (left_union.as_vector_double[1], right_union.as_vector_double[1]);
      temp1_union.as_vector_bool_long_long = vec_cmpge (left_union.as_vector_double[1], right_union.as_vector_double[1]);
      tempy_union.as_vector_double = vec_or (temp0_union.as_vector_double, temp1_union.as_vector_double);

      result_union.as_vector_double[0] = vec_nor (tempx_union.as_vector_double, tempx_union.as_vector_double);
      result_union.as_vector_double[1] = vec_nor (tempy_union.as_vector_double, tempy_union.as_vector_double);
    #endif
  }

  if ((condition & 7) == _CMP_NEQ_UQ)  /*100*/
  {
    #ifdef __ibmxl__
      temp0 = vec_cmpeq (left_union.as_vector_double[0], right_union.as_vector_double[0]);
      temp1 = vec_cmpeq (left_union.as_vector_double[1], right_union.as_vector_double[1]);
      result_union.as_vector_bool_long_long[0] = vec_nor (temp0, temp0);
      result_union.as_vector_bool_long_long[1] = vec_nor (temp1, temp1);
    #else
      temp0_union.as_vector_bool_long_long = vec_cmpeq (left_union.as_vector_double[0], right_union.as_vector_double[0]);
      temp1_union.as_vector_bool_long_long = vec_cmpeq (left_union.as_vector_double[1], right_union.as_vector_double[1]);
      result_union.as_vector_double[0] = vec_nor (temp0_union.as_vector_double, temp0_union.as_vector_double);
      result_union.as_vector_double[1] = vec_nor (temp1_union.as_vector_double, temp1_union.as_vector_double);
    #endif
  }
  if ((condition & 7) == _CMP_NLT_US)  /*101*/
  {
    #ifdef __ibmxl__
      temp0 = vec_cmplt (left_union.as_vector_double[0], right_union.as_vector_double[0]);
      temp1 = vec_cmplt (left_union.as_vector_double[1], right_union.as_vector_double[1]);
      result_union.as_vector_bool_long_long[0] = vec_nor (temp0, temp0);
      result_union.as_vector_bool_long_long[1] = vec_nor (temp1, temp1);
    #else
      temp0_union.as_vector_bool_long_long = vec_cmplt (left_union.as_vector_double[0], right_union.as_vector_double[0]);
      temp1_union.as_vector_bool_long_long = vec_cmplt (left_union.as_vector_double[1], right_union.as_vector_double[1]);
      result_union.as_vector_double[0] = vec_nor (temp0_union.as_vector_double, temp0_union.as_vector_double);
      result_union.as_vector_double[1] = vec_nor (temp1_union.as_vector_double, temp1_union.as_vector_double);
    #endif
  }
  if ((condition & 7) == _CMP_NLE_US)  /*110*/
  {
    #ifdef __ibmxl__
      temp0 = vec_cmple (left_union.as_vector_double[0], right_union.as_vector_double[0]);
      temp1 = vec_cmple (left_union.as_vector_double[1], right_union.as_vector_double[1]);
      result_union.as_vector_bool_long_long[0] = vec_nor (temp0, temp0);
      result_union.as_vector_bool_long_long[1] = vec_nor (temp1, temp1);
    #else
      temp0_union.as_vector_bool_long_long = vec_cmple (left_union.as_vector_double[0], right_union.as_vector_double[0]);
      temp1_union.as_vector_bool_long_long = vec_cmple (left_union.as_vector_double[1], right_union.as_vector_double[1]);
      result_union.as_vector_double[0] = vec_nor (temp0_union.as_vector_double, temp0_union.as_vector_double);
      result_union.as_vector_double[1] = vec_nor (temp1_union.as_vector_double, temp1_union.as_vector_double);
    #endif
  }
  if ((condition & 7) == _CMP_ORD_Q)  /*111*/
  {
    #ifdef __ibmxl__
      result_union.as_vector_bool_long_long[0] = vec_or (vec_cmplt (left_union.as_vector_double[0], right_union.as_vector_double[0]),
                                                            vec_cmpge (left_union.as_vector_double[0], right_union.as_vector_double[0]));
      result_union.as_vector_bool_long_long[1] = vec_or (vec_cmplt (left_union.as_vector_double[1], right_union.as_vector_double[1]),
                                                            vec_cmpge (left_union.as_vector_double[1], right_union.as_vector_double[1]));
    #else
      temp0_union.as_vector_bool_long_long = vec_cmplt (left_union.as_vector_double[0], right_union.as_vector_double[0]);
      temp1_union.as_vector_bool_long_long = vec_cmpge (left_union.as_vector_double[0], right_union.as_vector_double[0]);
      tempx_union.as_vector_double = vec_or (temp0_union.as_vector_double, temp1_union.as_vector_double);

      temp0_union.as_vector_bool_long_long = vec_cmplt (left_union.as_vector_double[1], right_union.as_vector_double[1]);
      temp1_union.as_vector_bool_long_long = vec_cmpge (left_union.as_vector_double[1], right_union.as_vector_double[1]);
      tempy_union.as_vector_double = vec_or (temp0_union.as_vector_double, temp1_union.as_vector_double);

      result_union.as_vector_double[0] = tempx_union.as_vector_double;
      result_union.as_vector_double[1] = tempy_union.as_vector_double;
    #endif
  }
  return result_union.as_m256d;
}
#endif

/************************************************************** Cast **************************************************/

/* Cast 2 64-bit doubles to 2 64-bit doubles and 2 undefined */
#ifdef VECLIB_VSX
VECLIB_INLINE __m256d vec_cast2dpto2dp2u (__m128d from)
{
  __m256d_union result_union;
  #ifdef __LITTLE_ENDIAN__
    result_union.as_m128d[0] = from;
    result_union.as_m128d[1] = (vector double) { 0., 0. }; /* undefined */
  #elif __BIG_ENDIAN__
    result_union.as_m128d[0] = (vector double) { 0., 0. }; /* undefined */
    result_union.as_m128d[1] = from;
  #endif
  return result_union.as_m256d;
}
#endif


/******************************************************** Unpack ******************************************************/

/* Unpack and interleave 2+2 64-bit doubles from high half of each 128-bit half */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m256d vec_unpackupper4dpto4dp (__m256d to_even, __m256d to_odd)
  {
    __m256_all_union to_even_union; to_even_union.as_m256d = to_even;
    __m256_all_union to_odd_union; to_odd_union.as_m256d = to_odd;
    __m256_all_union result_union;
    static const vector unsigned char permute_selector = {
      #ifdef __LITTLE_ENDIAN__
        0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,  0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F
      #elif __BIG_ENDIAN__
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,  0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07
      #endif
    };
    result_union.as_vector_unsigned_char[0] = vec_perm (to_even_union.as_vector_unsigned_char[0],
                                                        to_odd_union.as_vector_unsigned_char[0], permute_selector);
    result_union.as_vector_unsigned_char[1] = vec_perm (to_even_union.as_vector_unsigned_char[1],
                                                        to_odd_union.as_vector_unsigned_char[1], permute_selector);
    return result_union.as_m256d;
  }
#endif

/* Unpack and interleave 2+2 64-bit doubles from high half of each 128-bit half - deprecated - use previous function */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m256d vec_unpackupper44dpto4dp (__m256d to_even, __m256d to_odd)
  {
    return vec_unpackupper4dpto4dp (to_even, to_odd);
  }
#endif

/* Unpack and interleave 2+2 64-bit doubles from low half of each 128-bit half */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m256d vec_unpacklower4dpto4dp (__m256d to_even, __m256d to_odd)
  {
    __m256_all_union to_even_union; to_even_union.as_m256d = to_even;
    __m256_all_union to_odd_union; to_odd_union.as_m256d = to_odd;
    __m256_all_union result_union;
    static const vector unsigned char permute_selector = {
      #ifdef __LITTLE_ENDIAN__
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,  0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17
      #elif __BIG_ENDIAN__
        0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,  0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F
      #endif
    };
    result_union.as_vector_unsigned_char[0] = vec_perm (to_even_union.as_vector_unsigned_char[0],
                                                  to_odd_union.as_vector_unsigned_char[0], permute_selector);
    result_union.as_vector_unsigned_char[1] = vec_perm (to_even_union.as_vector_unsigned_char[1],
                                                  to_odd_union.as_vector_unsigned_char[1], permute_selector);
    return result_union.as_m256d;
  }
#endif

/* Unpack and interleave 2+2 64-bit doubles from low half of each 128-bit half - deprecated - use previous function */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m256d vec_unpacklower44dpto4dp (__m256d to_even, __m256d to_odd)
  {
    return vec_unpacklower4dpto4dp (to_even, to_odd);
  }
#endif

#endif
