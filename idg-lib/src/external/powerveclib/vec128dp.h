/******************************************************************************/
/*                                                                            */
/* Licensed Materials - Property of IBM                                       */
/*                                                                            */
/* IBM Power Vector Intrinisic Functions version 1.0.6                        */
/*                                                                            */
/* Copyright IBM Corp. 2014,2017                                              */
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

#ifndef _H_VEC128DP
#define _H_VEC128DP

#include <altivec.h>
#include "veclib_types.h"

/******************************************************** Load ********************************************************/

/* Load 2 64-bit doubles, unaligned */
VECLIB_INLINE __m128d vec_load2dpunaligned (double const* from) {
  #if __LITTLE_ENDIAN__
    /* LE Linux ABI *USED TO* require compilers to handle misaligned pointer dereferences. */
    return *((__m128d*) from);
  #elif __BIG_ENDIAN__
    #if (defined(__ibmxl__) && defined(__LITTLE_ENDIAN__))
      return (__m128d)vec_loadu1q((__m128i const* )from);
    #elif (defined(__ibmxl__) && defined(__BIG_ENDIAN__))
      return vec_permi(*((__m128d*)from),*((__m128d*)from), 2);
    #elif defined(__GNUC__)
      /* LE Linux ABI *USED TO* require compilers to handle misaligned pointer dereferences. */
      return *((__m128d*)from);
    #endif
  #endif
}

/* Load 64-bit double unaligned to lower part and zero upper part */
VECLIB_INLINE __m128d vec_loadlower1dpunaligned (double const* from) {
  #ifdef __LITTLE_ENDIAN__
    __m128d_union returnedVal;
    *(returnedVal.as_double) = *from;
    vector bool long long mask = (vector bool long long) { 0xFFFFFFFFFFFFFFFFull, 0x0000000000000000ull };
  #elif __BIG_ENDIAN__
    __m128d_union returnedVal;
    returnedVal.as_double[1] = *from;
    vector bool long long mask = (vector bool long long) { 0x0000000000000000ull, 0xFFFFFFFFFFFFFFFFull };
  #endif
  return vec_and(mask, returnedVal.as_m128d);
}

/* Load 64-bit double unaligned to upper part and zero lower part */
VECLIB_INLINE __m128d vec_loadupper1dpunaligned (__m128d a, double const* from) {
  #ifdef __LITTLE_ENDIAN__
    a[1] = *from;
    return a;
  #elif __BIG_ENDIAN__
    a[0] = *from;
    return a;
  #endif
}

/* Load 64-bit double and splat to 2 64-bit doubles */
VECLIB_INLINE __m128d vec_load1dp (double const* from) {
  #ifdef __ibmxl__
    __m128d doubleValue = (__m128d)vec_loadu1q( (__m128i*)from );
    return (__m128d)vec_mergeh( doubleValue, doubleValue );
  #else
    return vec_mergeh( *((__m128d*)from), *((__m128d*)from));
  #endif
}

/* Load 2 64-bit doubles, aligned */
VECLIB_INLINE __m128d vec_load2dpaligned (double const* from) {
  #ifdef __ibmxl__
    /* LE Linux ABI *USED TO* require compilers to handle misaligned pointer dereferences. */
    return *((__m128d*)from);
  #else
    #ifdef __LITTLE_ENDIAN__
      /* LE Linux ABI *USED TO* require compilers to handle misaligned pointer dereferences. */
      return *((__m128d*)from);
    #elif __BIG_ENDIAN__
      return vec_mergeh( *((__m128d*)(from+1)), *((__m128d*)(from)));
    #endif
  #endif
}

/******************************************************** Set *********************************************************/

/* Set 2 64-bit doubles to zero */
VECLIB_INLINE __m128d vec_zero2dp (void)
{
  return (__m128d) vec_splats (0);
}

/* Splat 64-bit double to 2 64-bit doubles */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m128d vec_splat2dp (double scalar)
  {
    return (__m128d) vec_splats (scalar);
  }
#endif

/* Set 2 64-bit doubles */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m128d vec_set2dp (double d1, double d0)
  {
    __m128d_union t;
    #ifdef __LITTLE_ENDIAN__
      t.as_double[0] = d0;
      t.as_double[1] = d1;
    #elif __BIG_ENDIAN__
      t.as_double[0] = d1;
      t.as_double[1] = d0;
    #endif
    return (__m128d) t.as_vector_double;
  }
#endif

/* Set 2 64-bit doubles reversed */
VECLIB_INLINE __m128d vec_setreverse2dp (double d1, double d0)
{
  return vec_set2dp (d0, d1);
}

/* Set lower 64-bit double and zero upper part */
VECLIB_INLINE __m128d vec_SetLower1dp (double d) {
  __m128d returnedVal;
  __m64_union zeroedDouble;
  zeroedDouble.as_long_long = 0x0ull;
  #ifdef __LITTLE_ENDIAN__
    returnedVal[0] = d;
    returnedVal[1] = zeroedDouble.as_double;
  #elif __BIG_ENDIAN__
    returnedVal[0] = zeroedDouble.as_double;
    returnedVal[1] = d;
  #endif
  return returnedVal;
}

/******************************************************* Store ********************************************************/

/* Store upper 64-bit double */
#ifdef VECLIB_VSX
  VECLIB_INLINE void vec_storeupper1dpof2dp (double* to, __m128d from)
  {
    __m128d_union from_union; from_union.as_m128d = from;
    unsigned int element_number;
    #ifdef __LITTLE_ENDIAN__
      element_number = 1;
    #elif __BIG_ENDIAN__
      element_number = 0;
    #endif
    *to = from_union.as_double[element_number];
  }
#endif

/* Store lower 64-bit double */
#ifdef VECLIB_VSX
  VECLIB_INLINE void vec_storelower1dpof2dp (double* to, __m128d from)
  {
    __m128d_union from_union; from_union.as_m128d = from;
    unsigned int element_number;
    #ifdef __LITTLE_ENDIAN__
      element_number = 0;
    #elif __BIG_ENDIAN__
      element_number = 1;
    #endif
    *to = from_union.as_double[element_number];
  }
#endif


/* Store 2 64-bit doubles, aligned */
VECLIB_INLINE void vec_store2dpto2dp (double* to, __m128d from) {
  #ifdef __ibmxl__
    #ifdef __LITTLE_ENDIAN__
      vec_xst(from, 0, to);
    #elif __BIG_ENDIAN__
      __m128d * to__m128d = (__m128d *) to;
      *to__m128d = from;
    #endif
  #elif (defined __GNUC__) && (__GCC_VERSION__ >= 492)
    vec_xst(from, 0, to);
  #elif (defined __GNUC__) && (__GCC_VERSION__ < 492)
    vec_st((vector float)from, 0, (float*)to);
  #else
    __m128d * to__m128d = (__m128d *) to;
    *to__m128d = from;
  #endif
}

/******************************************************* Insert *******************************************************/

/* Insert lower 64-bit double into lower */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m128d vec_insertlower1dp (__m128d into, __m128d from)
  {
    __m128_all_union into_union;
    __m128_all_union from_union;
    into_union.as_m128d = into;
    from_union.as_m128d = from;
    #ifdef __BIG_ENDIAN__
      static const vector unsigned char permute_selector = {
        0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07, 0x18,0x19,0x1A,0x1B,0x1C,0x1D,0x1E,0x1F
      };
      return (__m128d) vec_perm ((vector unsigned char) into, (vector unsigned char) from, permute_selector);
    #elif __LITTLE_ENDIAN__
      static const vector unsigned char permute_selector = {
        0x10,0x11,0x12,0x13,0x14,0x15,0x16,0x17, 0x08,0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F
      };
      return (__m128d) vec_perm ((vector unsigned char) into, (vector unsigned char) from, permute_selector);
    #endif
  }
#endif

/******************************************************** Extract *****************************************************/

/* Extract lower or upper half of 64+64-bit doubles pair */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m128d vec_extract2dpfrom4dp (__m256d from, const intlit1 element_number)
  {
    if ((element_number & 1) == 0)
    {
      #ifdef __LITTLE_ENDIAN__
        return from.m128d_0;
      #elif __BIG_ENDIAN__
        return from.m128d_1;
      #endif
    } else
    {
      #ifdef __LITTLE_ENDIAN__
        return from.m128d_1;
      #elif __BIG_ENDIAN__
        return from.m128d_0;
      #endif
    }
  }
#endif

/* Extract lower 64-bit double */
VECLIB_INLINE double vec_extractlowerdp (__m128d from) {
  #ifdef __LITTLE_ENDIAN__
    return from [0];
  #elif __BIG_ENDIAN__
    return from [1];
  #endif
}

/******************************************************* Permute ******************************************************/

/* Blend 2+2 64-bit doubles under mask to 2 64-bit doubles */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m128d vec_permute2dp (__m128d left, __m128d right, const intlit2 mask)
  {
    static const vector unsigned char permute_selector[4] = {
      /* to select left element for 0 or right element for 1 */
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
    return (__m128d) vec_perm ((vector unsigned char) left, (vector unsigned char) right, permute_selector[mask & 0x3]);
  }
#endif

/* Blend 2+2 64-bit doubles under mask to 2 64-bit doubles - deprecated - use previous function */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m128d vec_permute22dp (__m128d left, __m128d right, const intlit2 mask)
  {
    return vec_permute2dp (left, right, mask);
  }
#endif

/* Blend 2+2 64-bit doubles under mask to 2 64-bit doubles */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m128d vec_permutevr2dp (__m128d left, __m128d right, __m128d mask)
  {
    /* upper bit of each element mask selects 0 = left 1 = right */
    vector bool int select_mask = vec_cmplt ((vector signed int) mask, vec_splats (0));  /* convert upper bits to zeros or ones int mask */
    static const vector unsigned char permute_selector = {
      /* to copy upper word to second and third to lower */
      #ifdef __LITTLE_ENDIAN__
        0x0B,0x0A,0x09,0x08, 0x0B,0x0A,0x09,0x08, 0x03,0x02,0x01,0x00, 0x03,0x02,0x01,0x00
      #elif __BIG_ENDIAN__
        0x00,0x01,0x02,0x03, 0x00,0x01,0x02,0x03, 0x08,0x09,0x0A,0x0B, 0x08,0x09,0x0A,0x0B
      #endif
    };
    select_mask = vec_perm (select_mask, select_mask, permute_selector);
    return (__m128d) (vector double) vec_sel ((vector unsigned char) left, (vector unsigned char) right, (vector bool char) select_mask);
  }
#endif

/* Blend 2+2 64-bit doubles under mask to 2 64-bit doubles - deprecated - use previous function */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m128d vec_permutevr22dp (__m128d left, __m128d right, __m128d mask)
  {
    return vec_permutevr2dp (left, right, mask);
  }
#endif

/* Splat low 64-bit double to 2 64-bit doubles */
VECLIB_INLINE __m128d vec_extractlowertoupper (__m128d a) {
  #ifdef __LITTLE_ENDIAN__
    return vec_mergeh(a, a);
  #elif __BIG_ENDIAN__
    return vec_mergel(a, a);
  #endif
}

/********************************************* Convert integer to floating-point **************************************/

/* Convert lower 2 32-bit ints to 2 64-bit doubles */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m128d vec_convert2swto2dp (__m128i v)
  {
    __m128d_union t;
    __m128i_union ints;
    ints.as_m128i = v;
    #ifdef __LITTLE_ENDIAN__
      t.as_double[0] = (double) ints.as_int[0];
      t.as_double[1] = (double) ints.as_int[1];
    #elif __BIG_ENDIAN__
      t.as_double[0] = (double) ints.as_int[2];
      t.as_double[1] = (double) ints.as_int[3];
    #endif
    return t.as_m128d;
  }
#endif

/****************************************************** Arithmetic ****************************************************/

/* Add 2 64-bit doubles */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m128d vec_add2dp (__m128d left, __m128d right)
  {
    return (__m128d) vec_add ((vector double) left, (vector double) right);
  }
#endif

/* Subtract 2 64-bit doubles */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m128d vec_subtract2dp (__m128d left, __m128d right)
  {
    return (__m128d) vec_sub ((vector double) left, (vector double) right);
  }
#endif

/* Multiply 2 64-bit doubles */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m128d vec_multiply2dp (__m128d left, __m128d right)
  {
    return (__m128d) vec_mul ((vector double) left, (vector double) right);
  }
#endif

/* Multiply lower 64-bit doubles and insert */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m128d vec_multiplylower2dp (__m128d left, __m128d right)
  {
    __m128d_union t;
    __m128d_union leftx;
    leftx.as_m128d = left;

    #ifdef __BIG_ENDIAN__
      t.as_m128d = vec_mul ((vector double) left, (vector double) right);
      t.as_double[0] = leftx.as_double[0];
    #elif __LITTLE_ENDIAN__
      t.as_m128d = vec_mul ((vector double) left, (vector double) right);
      t.as_double[1] = leftx.as_double[1];
    #endif
    return t.as_m128d;
  }
#endif

/* Divide 2 64-bit doubles */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m128d vec_divide2dp (__m128d left, __m128d right)
  {
    return (__m128d) vec_div ((vector double) left, (vector double) right);
  }
#endif

/* Divide lower 64-bit doubles */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m128d vec_dividelower2dp (__m128d left, __m128d right)
  {
    __m128d_union t;
    __m128d_union leftx;
    leftx.as_m128d = left;

    t.as_m128d = vec_divide2dp (left, right);
    #ifdef __BIG_ENDIAN__
      t.as_double[0] = leftx.as_double[0];
    #elif __LITTLE_ENDIAN__
      t.as_double[1] = leftx.as_double[1];
    #endif
    return t.as_m128d;
  }
#endif

/* Square root of 2 64-bit doubles */
VECLIB_INLINE __m128d vec_squareroot2dp (__m128d v)
{
  return (__m128d) vec_sqrt (v);
}

/* Max 2 64-bit doubles */
VECLIB_INLINE __m128d vec_max2dp (__m128d left, __m128d right)
{
  return (__m128d) vec_max ((vector double) left, (vector double) right);
}

/* Min 2 64-bit doubles */
VECLIB_INLINE __m128d vec_min2dp (__m128d left, __m128d right)
{
  return (__m128d) vec_min ((vector double) left, (vector double) right);
}

/* Add lower 64-bit doubles and insert */
VECLIB_INLINE __m128d vec_AddLower1dp (__m128d left, __m128d right) {
  #ifdef __LITTLE_ENDIAN__
    vector bool long long mask = (vector bool long long) { 0xFFFFFFFFFFFFFFFFull, 0x0000000000000000ull };
  #elif __BIG_ENDIAN__
    vector bool long long mask = (vector bool long long) { 0x0000000000000000ull, 0xFFFFFFFFFFFFFFFFull };
  #endif
  return (__m128d)vec_add((__m128d)left, (__m128d)vec_and(right, mask));
}

/* Subtract lower 64-bit doubles and insert */
VECLIB_INLINE __m128d vec_SubLower1dp (__m128d left, __m128d right) {
  #ifdef __LITTLE_ENDIAN__
    vector bool long long mask = (vector bool long long) { 0xFFFFFFFFFFFFFFFFull, 0x0000000000000000ull };
  #elif __BIG_ENDIAN__
    vector bool long long mask = (vector bool long long) { 0x0000000000000000ull, 0xFFFFFFFFFFFFFFFFull };
  #endif
  return (__m128d)vec_sub((__m128d)left, (__m128d)vec_and(right, mask));
}

/* Max lower 64-bit doubles and insert */
VECLIB_INLINE __m128d vec_MaxLower1dp (__m128d left, __m128d right) {
  #ifdef __LITTLE_ENDIAN__
    vector bool long long mask = (vector bool long long) { 0xFFFFFFFFFFFFFFFFull, 0x0000000000000000ull };
  #elif __BIG_ENDIAN__
    vector bool long long mask = (vector bool long long) { 0x0000000000000000ull, 0xFFFFFFFFFFFFFFFFull };
  #endif
  return vec_or((vector bool long long)vec_andc(left, mask), vec_and(vec_max(left, right), mask));
}

/* Min lower 64-bit doubles and insert */
VECLIB_INLINE __m128d vec_MinLower1dp (__m128d left, __m128d right) {
  #ifdef __LITTLE_ENDIAN__
    vector bool long long mask = (vector bool long long) { 0xFFFFFFFFFFFFFFFFull, 0x0000000000000000ull };
  #elif __BIG_ENDIAN__
    vector bool long long mask = (vector bool long long) { 0x0000000000000000ull, 0xFFFFFFFFFFFFFFFFull };
  #endif
  return vec_or((vector bool long long)vec_andc(left, mask), vec_and(vec_min(left, right), mask));
}

/* Floor 2 64-bit doubles */
VECLIB_INLINE __m128d vec_Floor2dp (__m128d v) {
  return vec_floor(v);
}

/************************************************ Boolean *************************************************************/

/* Bitwise 128-bit and */
VECLIB_INLINE __m128d vec_bitand2dp (__m128d left, __m128d right)
{
  return (__m128d) vec_and ((vector double) left, (vector double) right);
}

/* Bitwise 128-bit andnot (reversed) */
VECLIB_INLINE __m128d vec_bitandnotleft2dp (__m128d left, __m128d right)
{
  return (__m128d) vec_andc ((vector double) right, (vector double) left);
}

/* Bitwise 128-bit or */
VECLIB_INLINE __m128d vec_bitor2dp (__m128d left, __m128d right)
{
  return (__m128d) vec_or ((vector double) left, (vector double) right);
}

/* Bitwise 128-bit xor */
VECLIB_INLINE __m128d vec_bitxor2dp (__m128d left, __m128d right)
{
  return (__m128d) vec_xor ((vector double) left, (vector double) right);
}

/******************************************************* Compare ******************************************************/

/* Compare eq */

/* Compare 2 64-bit doubles for == to mask */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m128d vec_compareeq2dp (__m128d left, __m128d right)
  {
    return (__m128d) vec_cmpeq ((vector double) left, (vector double) right);
  }
#endif

/* Compare lt */

/* Compare 2 64-bit doubles for < to mask */
VECLIB_INLINE __m128d vec_comparelt2dp (__m128d left, __m128d right)
{
  return (__m128d) vec_cmplt ((vector double) left, (vector double) right);
}

/* Compare scalar 64-bit doubles for != to mask */
VECLIB_INLINE __m128d vec_comparelower1dp (__m128d left, __m128d right) {
  #ifdef __LITTLE_ENDIAN__
    vector bool long long mask = (vector bool long long) { 0xFFFFFFFFFFFFFFFFull, 0x0000000000000000ull };
  #elif __BIG_ENDIAN__
    vector bool long long mask = (vector bool long long) { 0x0000000000000000ull, 0xFFFFFFFFFFFFFFFFull };
  #endif
  vector double ajfeieof = (vector double)vec_cmpeq((vector double)left, (vector double)right);
  return vec_or(
    (vector bool long long)vec_andc(left, mask),
    (__m128d)vec_andc(
      mask,
      ajfeieof)
  );
}

/* Compare le */

/* Compare 2 64-bit doubles for <= to mask */
VECLIB_INLINE __m128d vec_comparele2dp (__m128d left, __m128d right)
{
  return (__m128d) vec_cmple ((vector double) left, (vector double) right);
}

/* Compare gt */

/* Compare 2 64-bit doubles for > to mask */
VECLIB_INLINE __m128d vec_comparegt2dp (__m128d left, __m128d right)
{
  return (__m128d) vec_cmpgt ((vector double) left, (vector double) right);
}

/* Compare ge */

/* Compare 2 64-bit doubles for >= to mask */
VECLIB_INLINE __m128d vec_comparege2dp (__m128d left, __m128d right)
{
  return (__m128d) vec_cmpge ((vector double) left, (vector double) right);
}

/* Compare not */

/* Compare 2 64-bit doubles for != to mask */
VECLIB_INLINE __m128d vec_comparenoteq2dp (__m128d left, __m128d right)
{
  __m128d_union leftx;
  leftx.as_m128d = vec_compareeq2dp (left, right);
  __m128d_union rightx;
  rightx.as_m128d = (__m128d) vec_splats ((unsigned char) 0xFF);

  return (__m128d) vec_xor ((vector double) leftx.as_m128d, (vector double) rightx.as_m128d);
}

/* Compare 2 64-bit doubles for !>= to mask */
VECLIB_INLINE __m128d vec_comparenotge2dp (__m128d left, __m128d right)
{
  __m128d_union leftx;
  leftx.as_m128d = vec_comparege2dp (left, right);
  __m128d_union rightx;
  rightx.as_m128d = (__m128d) vec_splats ((unsigned char) 0xFF);

  return (__m128d) vec_xor ((vector double) leftx.as_m128d, (vector double) rightx.as_m128d);
}

/* Compare 2 64-bit doubles for !> to mask */
VECLIB_INLINE __m128d vec_comparenotgt2dp (__m128d left, __m128d right)
{
  __m128d_union leftx;
  leftx.as_m128d = vec_comparegt2dp (left, right);
  __m128d_union rightx;
  rightx.as_m128d = (__m128d) vec_splats ((unsigned char) 0xFF);

  return (__m128d) vec_xor ((vector double) leftx.as_m128d, (vector double) rightx.as_m128d);
}

/* Compare 2 64-bit doubles for !<= to mask */
VECLIB_INLINE __m128d vec_comparenotle2dp (__m128d left, __m128d right)
{
  __m128d_union leftx;
  leftx.as_m128d = vec_comparele2dp (left, right);
  __m128d_union rightx;
  rightx.as_m128d = (__m128d) vec_splats ((unsigned char) 0xFF);

  return (__m128d) vec_xor ((vector double) leftx.as_m128d, (vector double) rightx.as_m128d);
}

/* Compare 2 64-bit doubles for !< to mask */
VECLIB_INLINE __m128d vec_comparenotlt2dp (__m128d left, __m128d right)
{
  __m128d_union leftx;
  leftx.as_m128d = vec_comparelt2dp (left, right);
  __m128d_union rightx;
  rightx.as_vector_unsigned_long_long =  vec_splats (0xFFFFFFFFFFFFFFFFULL);
  __m128d_union resultx;
  resultx.as_vector_double = vec_xor (leftx.as_vector_double, rightx.as_vector_double);
  return resultx.as_m128d;
}

/* Compare 2 64-bit doubles for not NaNs to mask */
VECLIB_INLINE __m128d vec_comparenotnans2dp (__m128d left, __m128d right)
{
  __m128d_union leftx;
  __m128d_union rightx;
  leftx.as_m128d = vec_compareeq2dp(left, left);
  rightx.as_m128d = vec_compareeq2dp(right, right);
  return (__m128d) vec_and (leftx.as_vector_double, rightx.as_vector_double);
}

/* Compare 2 64-bit doubles for NaNs to mask */
VECLIB_INLINE __m128d vec_comparenans2dp (__m128d left, __m128d right)
{
  __m128d_union leftx;
  leftx.as_m128d = vec_comparenotnans2dp (left, right);
  __m128d_union rightx;
  rightx.as_vector_unsigned_long_long = vec_splats (0xFFFFFFFFFFFFFFFFULL);
  __m128d_union resultx;
  resultx.as_vector_double = vec_xor (leftx.as_vector_double, rightx.as_vector_double);
  return resultx.as_m128d;
}
/* Compare 1 64-bit double for == to mask */
VECLIB_INLINE __m128d vec_compareeqlower1of2dp (__m128d left, __m128d right) {
  __m128d_union inter_union;
    __m128d_union res_union;
  vector unsigned long long select_vector = {
    0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL
  };
  inter_union.as_m128d = vec_compareeq2dp ((vector double) left, (vector double) right);
  res_union.as_vector_double = vec_sel ((vector double) left, inter_union.as_vector_double, select_vector);
  return res_union.as_m128d;
}

/* Compare 1 64-bit double for >= to mask */
VECLIB_INLINE __m128d vec_comparegelower1of2dp (__m128d left, __m128d right) {
  __m128d_union inter_union;
  __m128d_union res_union;
  vector unsigned long long select_vector = {
    0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL
  };
  inter_union.as_m128d = vec_comparege2dp ((vector double) left, (vector double) right);
  res_union.as_vector_double = vec_sel ((vector double) left, inter_union.as_vector_double, select_vector);
  return res_union.as_m128d;
}

/* Compare 1 64-bit double for > to mask */
VECLIB_INLINE __m128d vec_comparegtlower1of2dp (__m128d left, __m128d right) {
  __m128d_union inter_union;
  __m128d_union res_union;
  vector unsigned long long select_vector = {
    0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL
  };
  inter_union.as_m128d = vec_comparegt2dp ((vector double) left, (vector double) right);
  res_union.as_vector_double = vec_sel ((vector double) left, inter_union.as_vector_double, select_vector);
  return res_union.as_m128d;
}

/* Compare 1 64-bit double for <= to mask */
VECLIB_INLINE __m128d vec_comparelelower1of2dp (__m128d left, __m128d right) {
  __m128d_union inter_union;
  __m128d_union res_union;
  vector unsigned long long select_vector = {
    0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL
  };
  inter_union.as_m128d = vec_comparele2dp ((vector double) left, (vector double) right);
  res_union.as_vector_double = vec_sel ((vector double) left, inter_union.as_vector_double, select_vector);
  return res_union.as_m128d;
}

/* Compare 1 64-bit double for < to mask */
VECLIB_INLINE __m128d vec_compareltlower1of2dp (__m128d left, __m128d right) {
  __m128d_union inter_union; 
  __m128d_union res_union;
  
  vector unsigned long long select_vector = {
    0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL
  };
  inter_union.as_m128d = vec_comparelt2dp ( left,  right);
  res_union.as_vector_double = vec_sel ((vector double) left, inter_union.as_vector_double, select_vector);
  return res_union.as_m128d;
}

/* Compare 1 64-bit double for !>= to mask */
VECLIB_INLINE __m128d vec_comparengelower1of2dp (__m128d left, __m128d right) {
  __m128d_union inter_union;
  __m128d_union res_union;
  __m128d_union select_union;
  select_union.as_unsigned_long_hex[0] = 0xFFFFFFFFFFFFFFFFULL;
  select_union.as_unsigned_long_hex[1] = 0x0000000000000000ULL;
  
  inter_union.as_m128d = vec_comparenotge2dp (left, right);
  res_union.as_vector_double = vec_sel ((vector double) left, inter_union.as_vector_double, select_union.as_vector_unsigned_long_long);
  return res_union.as_m128d;
}

/* Compare 1 64-bit double for !> to mask */
VECLIB_INLINE __m128d vec_comparengtlower1of2dp (__m128d left, __m128d right) {
  __m128d_union inter_union;
  __m128d_union res_union;
  vector unsigned long long select_vector = {
    0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL
  };
  inter_union.as_m128d = vec_comparenotgt2dp ( left,  right);
  res_union.as_vector_double = vec_sel ((vector double) left, inter_union.as_vector_double, select_vector);
  return res_union.as_m128d;
}

/* Compare 1 64-bit double for !<= to mask */
VECLIB_INLINE __m128d vec_comparenlelower1of2dp (__m128d left, __m128d right) {
  __m128d_union inter_union;
  __m128d_union res_union;
  vector unsigned long long select_vector = {
    0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL
  };
  inter_union.as_m128d = vec_comparenotle2dp ( left,  right);
  res_union.as_vector_double = vec_sel ((vector double) left, inter_union.as_vector_double, select_vector);
  return res_union.as_m128d;
}

/* Compare 1 64-bit double for !< to mask */
VECLIB_INLINE __m128d vec_comparenltlower1of2dp (__m128d left, __m128d right) {
  __m128d_union inter_union;
  __m128d_union res_union;
  vector unsigned long long select_vector = {
    0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL
  };
  inter_union.as_m128d = vec_comparenotlt2dp ( left,  right);
  res_union.as_vector_double = vec_sel ((vector double) left, inter_union.as_vector_double, select_vector);
  return res_union.as_m128d;
}

/* Compare 1 64-bit double for not NaNs to mask */
VECLIB_INLINE __m128d vec_comparenotnanslower1of2dp (__m128d left, __m128d right) {
  __m128d_union inter_union;
  __m128d_union res_union;
  vector unsigned long long select_vector = {
    0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL
  };
  inter_union.as_m128d = vec_comparenotnans2dp ( left, right);
  res_union.as_vector_double = vec_sel ((vector double) left, inter_union.as_vector_double, select_vector);
  return res_union.as_m128d;
}

/* Compare 1 64-bit double for NaNs to mask */
VECLIB_INLINE __m128d vec_comparenanslower1of2dp (__m128d left, __m128d right) {
  __m128d_union inter_union;
  __m128d_union res_union;
  vector unsigned long long select_vector = {
    0xFFFFFFFFFFFFFFFFULL, 0x0000000000000000ULL
  };
  inter_union.as_m128d = vec_comparenans2dp ( left, right);
  res_union.as_vector_double = vec_sel ((vector double) left, inter_union.as_vector_double, select_vector);
  return res_union.as_m128d;
}

/* Compare 1 64-bit double for == to bool */
VECLIB_INLINE int vec_compareeqtoboollower1of2dp (__m128d left, __m128d right)
{
  __m128d_union res_union;
  __m128d_union nan_union;
  res_union.as_m128d = vec_compareeq2dp(left, right);
  nan_union.as_m128d = vec_comparenotnans2dp(left, right);
  /* Return 1s for NaN */
  return nan_union.as_unsigned_long_hex[0] == 0x0000000000000000ULL || res_union.as_unsigned_long_hex[0] == 0xFFFFFFFFFFFFFFFFULL;
}

/* Compare 1 64-bit double for >= to bool */
VECLIB_INLINE int vec_comparegetoboollower1of2dp (__m128d left, __m128d right){
  __m128d_union res_union;
  __m128d_union nan_union;
  res_union.as_m128d = vec_comparege2dp(left, right);
  nan_union.as_m128d = vec_comparenotnans2dp(left, right);
  /* Return 0s for NaN */
  return nan_union.as_unsigned_long_hex[0] == 0xFFFFFFFFFFFFFFFFULL &&  res_union.as_unsigned_long_hex[0] == 0xFFFFFFFFFFFFFFFFULL;
}

/* Compare 1 64-bit double for > to bool */
VECLIB_INLINE int vec_comparegttoboollower1of2dp (__m128d left, __m128d right){
  __m128d_union res_union;
  __m128d_union nan_union;
  res_union.as_m128d = vec_comparegt2dp(left, right);
  nan_union.as_m128d = vec_comparenotnans2dp(left, right);
  /* Return 0s for NaN */
  return nan_union.as_unsigned_long_hex[0] == 0xFFFFFFFFFFFFFFFFULL && res_union.as_unsigned_long_hex[0] == 0xFFFFFFFFFFFFFFFFULL;
}

/* Compare 1 64-bit double for <= to bool */
VECLIB_INLINE int vec_compareletoboollower1of2dp (__m128d left, __m128d right){
  __m128d_union res_union;
  __m128d_union nan_union;
  res_union.as_m128d = vec_comparele2dp(left, right);
  nan_union.as_m128d = vec_comparenotnans2dp(left, right);
  /* Return 1s for NaN */
  return nan_union.as_unsigned_long_hex[0] == 0x0000000000000000ULL || res_union.as_unsigned_long_hex[0] == 0xFFFFFFFFFFFFFFFFULL;
}

/* Compare 1 64-bit double for < to bool */
VECLIB_INLINE int vec_comparelttoboollower1of2dp (__m128d left, __m128d right){
  __m128d_union res_union;
  __m128d_union nan_union;
  res_union.as_m128d = vec_comparelt2dp(left, right);
  nan_union.as_m128d = vec_comparenotnans2dp(left, right);
  /* Return 1s for NaN */
  return nan_union.as_unsigned_long_hex[0] == 0x0000000000000000ULL || res_union.as_unsigned_long_hex[0] == 0xFFFFFFFFFFFFFFFFULL;
}

/* Compare 1 64-bit double for != to bool */
VECLIB_INLINE int vec_compareneqtoboollower1of2dp (__m128d left, __m128d right){
  __m128d_union res_union;
  __m128d_union nan_union;
  res_union.as_m128d = vec_comparenoteq2dp(left, right);
  nan_union.as_m128d = vec_comparenotnans2dp(left, right);
  /* Return 0s for NaN */
  return nan_union.as_unsigned_long_hex[0] == 0xFFFFFFFFFFFFFFFFULL && res_union.as_unsigned_long_hex[0] == 0xFFFFFFFFFFFFFFFFULL;
}

/* Compare 1 64-bit double for == to bool QNaNs nonsignaling */
VECLIB_INLINE int vec_compareeqtoboolnonsignalinglower1of2dp (__m128d left, __m128d right){
  __m128d_union res_union;
  __m128d_union nan_union;
  res_union.as_m128d = vec_compareeq2dp(left, right);
  nan_union.as_m128d = vec_comparenotnans2dp(left, right);
  /* Return 1s for NaN */
  return nan_union.as_unsigned_long_hex[0] == 0x0000000000000000ULL || res_union.as_unsigned_long_hex[0] == 0xFFFFFFFFFFFFFFFFULL;
}

/* Compare 1 64-bit double for >= to bool QNaNs nonsignaling */
VECLIB_INLINE int vec_comparegetoboolnonsignalinglower1of2dp (__m128d left, __m128d right){
  __m128d_union res_union;
  __m128d_union nan_union;
  res_union.as_m128d = vec_comparege2dp(left, right);
  nan_union.as_m128d = vec_comparenotnans2dp(left, right);
  /* Return 0s for NaN */
  return nan_union.as_unsigned_long_hex[0] == 0xFFFFFFFFFFFFFFFFULL &&  res_union.as_unsigned_long_hex[0] == 0xFFFFFFFFFFFFFFFFULL;
}

/* Compare 1 64-bit double for > to bool QNaNs nonsignaling */
VECLIB_INLINE int vec_comparegttoboolnonsignalinglower1of2dp (__m128d left, __m128d right){
  __m128d_union res_union;
  __m128d_union nan_union;
  res_union.as_m128d = vec_comparegt2dp(left, right);
  nan_union.as_m128d = vec_comparenotnans2dp(left, right);
  /* Return 0s for NaN */
  return nan_union.as_unsigned_long_hex[0] == 0xFFFFFFFFFFFFFFFFULL && res_union.as_unsigned_long_hex[0] == 0xFFFFFFFFFFFFFFFFULL;
}

/* Compare 1 64-bit double for <= to bool QNaNs nonsignaling */
VECLIB_INLINE int vec_compareletoboolnonsignalinglower1of2dp (__m128d left, __m128d right){
  __m128d_union res_union;
  __m128d_union nan_union;
  res_union.as_m128d = vec_comparele2dp(left, right);
  nan_union.as_m128d = vec_comparenotnans2dp(left, right);
  /* Return 1s for NaN */
  return nan_union.as_unsigned_long_hex[0] == 0x0000000000000000ULL || res_union.as_unsigned_long_hex[0] == 0xFFFFFFFFFFFFFFFFULL;
}

/* Compare 1 64-bit double for < to bool QNaNs nonsignaling */
VECLIB_INLINE int vec_comparelttoboolnonsignalinglower1of2dp (__m128d left, __m128d right){
  __m128d_union res_union;
  __m128d_union nan_union;
  res_union.as_m128d = vec_comparelt2dp(left, right);
  nan_union.as_m128d = vec_comparenotnans2dp(left, right);
  /* Return 1s for NaN */
  return nan_union.as_unsigned_long_hex[0] == 0x0000000000000000ULL || res_union.as_unsigned_long_hex[0] == 0xFFFFFFFFFFFFFFFFULL;
}

/* Compare 1 64-bit double for != to bool QNaNs nonsignaling */
VECLIB_INLINE int vec_compareneqtoboolnonsignalinglower1of2dp (__m128d left, __m128d right){
  __m128d_union res_union;
  __m128d_union nan_union;
  res_union.as_m128d = vec_comparenoteq2dp(left, right);
  nan_union.as_m128d = vec_comparenotnans2dp(left, right);
  /* Return 0s for NaN */
  return nan_union.as_unsigned_long_hex[0] == 0xFFFFFFFFFFFFFFFFULL && res_union.as_unsigned_long_hex[0] == 0xFFFFFFFFFFFFFFFFULL;
}

/****************************************************** Type Cast *****************************************************/

/* Cast type __m128i to __m128d */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m128d vec_cast1qto2dp (__m128i v)
  {
    __m128_all_union v_union;
    v_union.as_m128i = v;
    return (__m128d) v_union.as_m128d;
  }
#endif

/* Cast lower 2 of 4 64-bit doubles to 2 64-bit doubles */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m128d vec_cast2of4dpto2dp (__m256d from)
  {
    __m256d_union from_union;  from_union.as_m256d = from;
  #ifdef __LITTLE_ENDIAN__
      return from_union.as_m128d[0];
  #elif __BIG_ENDIAN__
    return from_union.as_m128d[1];
  #endif
  }
#endif

/* Cast __m128 to __m128d */
VECLIB_INLINE __m128d vec_Cast4spto2dp (__m128 from) {
  return (__m128d)from;
  __m128_all_union newFrom; newFrom.as_m128 = from;
  return newFrom.as_m128d;
}

/**************************************************** Mathematics *****************************************************/

/* Add 1 odd and subtract 1 even of 2 64-bit doubles - (A0-B0, A1+B1) */
VECLIB_INLINE __m128d vec_addsub2dp (__m128d left, __m128d right) {
  __m128_all_union negation;
  negation.as_vector_unsigned_int = (vector unsigned int) {
    #ifdef __LITTLE_ENDIAN__
      0x00000000u, 0x80000000u, 0x00000000u, 0x00000000u
    #elif __BIG_ENDIAN__
      0x00000000u, 0x00000000u, 0x80000000u, 0x00000000u
    #endif
  };
  __m128d tempResult = (vector double)vec_xor( right , negation.as_vector_double);
  return (vector double)vec_add(left, tempResult);
}

/* Horizontally add 1+1 adjacent pairs of 64-bit doubles to 2 64-bit doubles - (B1+B0, A1+A0) */
VECLIB_INLINE __m128d  vec_horizontaladd2dp (__m128d lower, __m128d upper) {
  #ifdef __LITTLE_ENDIAN__
    return (vector double)vec_add(vec_mergeh(lower, upper), vec_mergel(lower, upper));
  #elif __BIG_ENDIAN__
    return (vector double)vec_add(vec_mergeh(upper, lower ), vec_mergel( upper, lower));
  #endif
}

/* Horizontally subtract lower pair then upper pair of 64-bit doubles - (A0-A1, B0-B1) */
VECLIB_INLINE __m128d vec_horizontalsub2dp (__m128d lower, __m128d upper) {
  #ifdef __LITTLE_ENDIAN__
    return (vector double)vec_sub(vec_mergeh( lower, upper ), vec_mergel( lower, upper ));
  #elif __BIG_ENDIAN__
    return (vector double)vec_sub(vec_mergel( upper, lower ), vec_mergeh( upper, lower ));
  #endif
}

/************************************************* Unpack *************************************************************/

/* Unpack 1+1 64-bit doubles from high halves and interleave */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m128d vec_unpackupper1dpto2dp (__m128d to_even, __m128d to_odd)
  {
    __m128_all_union to_even_union; to_even_union.as_m128d = to_even;
    __m128_all_union to_odd_union; to_odd_union.as_m128d = to_odd;
    __m128_all_union result_union;
    static const vector unsigned char permute_selector = {
      #ifdef __LITTLE_ENDIAN__
        0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,  0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F
      #elif __BIG_ENDIAN__
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,  0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07
      #endif
    };
    result_union.as_vector_unsigned_char = vec_perm (to_even_union.as_vector_unsigned_char,
                                                     to_odd_union.as_vector_unsigned_char, permute_selector);
    return result_union.as_m128d;
  }
#endif

/* Unpack 1+1 64-bit doubles from high halves and interleave - deprecated - use previous function */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m128d vec_unpackupper11dpto2dp (__m128d to_even, __m128d to_odd)
  {
    return vec_unpackupper1dpto2dp (to_even, to_odd);
  }
#endif

/* Unpack 1+1 64-bit doubles from low halves and interleave */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m128d vec_unpacklower1dpto2dp (__m128d to_even, __m128d to_odd)
  {
    __m128_all_union to_even_union; to_even_union.as_m128d = to_even;
    __m128_all_union to_odd_union; to_odd_union.as_m128d = to_odd;
    __m128_all_union result_union;
    static const vector unsigned char permute_selector = {
      #ifdef __LITTLE_ENDIAN__
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,  0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17
      #elif __BIG_ENDIAN__
        0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,  0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F
      #endif
    };
    result_union.as_vector_unsigned_char = vec_perm (to_even_union.as_vector_unsigned_char,
                                                     to_odd_union.as_vector_unsigned_char, permute_selector);
    return result_union.as_m128d;
  }
#endif

/* Unpack 1+1 64-bit doubles from low halves and interleave - deprecated - use previous function */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m128d vec_unpacklower11dpto2dp (__m128d to_even, __m128d to_odd)
  {
    return vec_unpacklower1dpto2dp (to_even, to_odd);
  }
#endif

#endif
