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

#ifndef _H_VEC128SP
#define _H_VEC128SP

#include <altivec.h>
#include "veclib_types.h"


/******************************************************************************/

/*- Needed for vec_dotproduct4sp */
static const vector bool int expand_bit_to_word_masks[16] = {
  #ifdef __LITTLE_ENDIAN__
    { 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u }, /* elements 0000 */
    { 0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u }, /* elements 0001 */
    { 0x00000000u, 0xFFFFFFFFu, 0x00000000u, 0x00000000u }, /* elements 0010 */
    { 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u, 0x00000000u }, /* elements 0011 */
    { 0x00000000u, 0x00000000u, 0xFFFFFFFFu, 0x00000000u }, /* elements 0100 */
    { 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu, 0x00000000u }, /* elements 0101 */
    { 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u }, /* elements 0110 */
    { 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u }, /* elements 0111 */
    { 0x00000000u, 0x00000000u, 0x00000000u, 0xFFFFFFFFu }, /* elements 1000 */
    { 0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0xFFFFFFFFu }, /* elements 1001 */
    { 0x00000000u, 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu }, /* elements 1010 */
    { 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu }, /* elements 1011 */
    { 0x00000000u, 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu }, /* elements 1100 */
    { 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu }, /* elements 1101 */
    { 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu }, /* elements 1110 */
    { 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu }  /* elements 1111 */
  #elif __BIG_ENDIAN__
    { 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u }, /* elements 0000 */
    { 0x00000000u, 0x00000000u, 0x00000000u, 0xFFFFFFFFu }, /* elements 0001 */
    { 0x00000000u, 0x00000000u, 0xFFFFFFFFu, 0x00000000u }, /* elements 0010 */
    { 0x00000000u, 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu }, /* elements 0011 */
    { 0x00000000u, 0xFFFFFFFFu, 0x00000000u, 0x00000000u }, /* elements 0100 */
    { 0x00000000u, 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu }, /* elements 0101 */
    { 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u }, /* elements 0110 */
    { 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu }, /* elements 0111 */
    { 0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u }, /* elements 1000 */
    { 0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0xFFFFFFFFu }, /* elements 1001 */
    { 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu, 0x00000000u }, /* elements 1010 */
    { 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu, 0xFFFFFFFFu }, /* elements 1011 */
    { 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u, 0x00000000u }, /* elements 1100 */
    { 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u, 0xFFFFFFFFu }, /* elements 1101 */
    { 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0x00000000u }, /* elements 1110 */
    { 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu }  /* elements 1111 */
  #endif
};

static const vector unsigned char permute_highest_word_to_words_masks[16] = {
  /*- Select from right vector, with left vector all zeros. */
  #ifdef __LITTLE_ENDIAN__
    { 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00 }, /* elements 0000 */
    { 0x13,0x12,0x11,0x10, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00 }, /* elements 0001 */
    { 0x00,0x00,0x00,0x00, 0x13,0x12,0x11,0x10, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00 }, /* elements 0010 */
    { 0x13,0x12,0x11,0x10, 0x13,0x12,0x11,0x10, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00 }, /* elements 0011 */
    { 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x13,0x12,0x11,0x10, 0x00,0x00,0x00,0x00 }, /* elements 0100 */
    { 0x13,0x12,0x11,0x10, 0x00,0x00,0x00,0x00, 0x13,0x12,0x11,0x10, 0x00,0x00,0x00,0x00 }, /* elements 0101 */
    { 0x00,0x00,0x00,0x00, 0x13,0x12,0x11,0x10, 0x13,0x12,0x11,0x10, 0x00,0x00,0x00,0x00 }, /* elements 0110 */
    { 0x13,0x12,0x11,0x10, 0x13,0x12,0x11,0x10, 0x13,0x12,0x11,0x10, 0x00,0x00,0x00,0x00 }, /* elements 0111 */
    { 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x13,0x12,0x11,0x10 }, /* elements 1000 */
    { 0x13,0x12,0x11,0x10, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x13,0x12,0x11,0x10 }, /* elements 1001 */
    { 0x00,0x00,0x00,0x00, 0x13,0x12,0x11,0x10, 0x00,0x00,0x00,0x00, 0x13,0x12,0x11,0x10 }, /* elements 1010 */
    { 0x13,0x12,0x11,0x10, 0x13,0x12,0x11,0x10, 0x00,0x00,0x00,0x00, 0x13,0x12,0x11,0x10 }, /* elements 1011 */
    { 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x13,0x12,0x11,0x10, 0x13,0x12,0x11,0x10 }, /* elements 1100 */
    { 0x13,0x12,0x11,0x10, 0x00,0x00,0x00,0x00, 0x13,0x12,0x11,0x10, 0x13,0x12,0x11,0x10 }, /* elements 1101 */
    { 0x00,0x00,0x00,0x00, 0x13,0x12,0x11,0x10, 0x13,0x12,0x11,0x10, 0x13,0x12,0x11,0x10 }, /* elements 1110 */
    { 0x13,0x12,0x11,0x10, 0x13,0x12,0x11,0x10, 0x13,0x12,0x11,0x10, 0x13,0x12,0x11,0x10 }  /* elements 1111 */
  #elif __BIG_ENDIAN__
    { 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00 }, /* elements 0000 */
    { 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x10,0x11,0x12,0x13 }, /* elements 0001 */
    { 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x10,0x11,0x12,0x13, 0x00,0x00,0x00,0x00 }, /* elements 0010 */
    { 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x10,0x11,0x12,0x13, 0x10,0x11,0x12,0x13 }, /* elements 0011 */
    { 0x00,0x00,0x00,0x00, 0x10,0x11,0x12,0x13, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00 }, /* elements 0100 */
    { 0x00,0x00,0x00,0x00, 0x10,0x11,0x12,0x13, 0x00,0x00,0x00,0x00, 0x10,0x11,0x12,0x13 }, /* elements 0101 */
    { 0x00,0x00,0x00,0x00, 0x10,0x11,0x12,0x13, 0x10,0x11,0x12,0x13, 0x00,0x00,0x00,0x00 }, /* elements 0110 */
    { 0x00,0x00,0x00,0x00, 0x10,0x11,0x12,0x13, 0x10,0x11,0x12,0x13, 0x10,0x11,0x12,0x13 }, /* elements 0111 */
    { 0x11,0x11,0x12,0x13, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00 }, /* elements 1000 */
    { 0x11,0x11,0x12,0x13, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00, 0x10,0x11,0x12,0x13 }, /* elements 1001 */
    { 0x11,0x11,0x12,0x13, 0x00,0x00,0x00,0x00, 0x10,0x11,0x12,0x13, 0x00,0x00,0x00,0x00 }, /* elements 1010 */
    { 0x11,0x11,0x12,0x13, 0x00,0x00,0x00,0x00, 0x10,0x11,0x12,0x13, 0x10,0x11,0x12,0x13 }, /* elements 1011 */
    { 0x11,0x11,0x12,0x13, 0x10,0x11,0x12,0x13, 0x00,0x00,0x00,0x00, 0x00,0x00,0x00,0x00 }, /* elements 1100 */
    { 0x11,0x11,0x12,0x13, 0x10,0x11,0x12,0x13, 0x00,0x00,0x00,0x00, 0x10,0x11,0x12,0x13 }, /* elements 1101 */
    { 0x11,0x11,0x12,0x13, 0x10,0x11,0x12,0x13, 0x10,0x11,0x12,0x13, 0x00,0x00,0x00,0x00 }, /* elements 1110 */
    { 0x11,0x11,0x12,0x13, 0x10,0x11,0x12,0x13, 0x10,0x11,0x12,0x13, 0x10,0x11,0x12,0x13 }  /* elements 1111 */
  #endif
};


/******************************************************** Load ********************************************************/

/* Load 4 32-bit floats, aligned */
VECLIB_INLINE __m128 vec_load4sp (float const* address)
{
  return (__m128) vec_ld (0, (vector float*)address);
}

/* Load 2 32-bit floats into upper 2 32-bit floats, unaligned */
VECLIB_INLINE __m128 vec_loaduandinsertupper2spinto4sp (__m128 a, __m64 const* from)
{
  __m64_union from_union; from_union.as_m64 = *from;
  __m128_union a_union; a_union.as_m128 = a;
  __m128_union result_union;
  #ifdef __LITTLE_ENDIAN__
    result_union.as_m64[0] = a_union.as_m64[0];
    result_union.as_float[2] = from_union.as_float[0];
    result_union.as_float[3] = from_union.as_float[1];
  #elif __BIG_ENDIAN__
    result_union.as_m64[1] = a_union.as_m64[1];
    result_union.as_float[0] = from_union.as_float[0];
    result_union.as_float[1] = from_union.as_float[1];
  #endif
  return result_union.as_m128;
}

/* Load 2 32-bit floats into upper 2 32-bit floats, unaligned - deprecated - use previous function */
VECLIB_INLINE __m128 vec_loadandinsertupper2spinto4sp (__m128 a, __m64 const* from)
{
  return vec_loaduandinsertupper2spinto4sp (a, from);
}

/* Load 2 32-bit floats into lower 2 32-bit floats, unaligned */
VECLIB_INLINE __m128 vec_loaduandinsertlower2spinto4sp (__m128 a, __m64 const* from)
{
  __m64_union from_union; from_union.as_m64 = *from;
  __m128_union a_union; a_union.as_m128 = a;
  __m128_union result_union;
  #ifdef __LITTLE_ENDIAN__
    result_union.as_float[0] = from_union.as_float[0];
    result_union.as_float[1] = from_union.as_float[1];
    result_union.as_m64[1] = a_union.as_m64[1];
  #elif __BIG_ENDIAN__
    result_union.as_float[2] = from_union.as_float[0];
    result_union.as_float[3] = from_union.as_float[1];
    result_union.as_m64[0] = a_union.as_m64[0];
  #endif
  return result_union.as_m128;
}

/* Load 2 32-bit floats into lower 2 32-bit floats, unaligned - deprecated - use previous function */
VECLIB_INLINE __m128 vec_loadandinsertlower2spinto4sp (__m128 a, __m64 const* from)
{
  return vec_loaduandinsertlower2spinto4sp (a, from);
}

/* Load scalar 32-bit float and zero upper 3 32-bit floats, unaligned */
VECLIB_INLINE __m128 vec_loaduzero1sp3z (float const* from)
{
  __m128_union result_union;
  result_union.as_m128 = (__m128) vec_splats ((float) 0.0);
  #ifdef __LITTLE_ENDIAN__
    result_union.as_float[0] = *from;
  #elif __BIG_ENDIAN__
    result_union.as_float[3] = *from;
  #endif
  return result_union.as_m128;
}

/* Load scalar 32-bit float and zero upper 3 32-bit floats, unaligned - deprecated - use previous function */
VECLIB_INLINE __m128 vec_loadzero1sp3zu (float const* from)
{
  return vec_loaduzero1sp3z (from);
}

/* Load 4 32-bit floats, unaligned */
VECLIB_INLINE __m128 vec_loadu4sp (float const* from)
{
  #if __LITTLE_ENDIAN__
    /* LE Linux ABI *USED TO* require compilers to handle misaligned pointer dereferences. */
    return (__m128) *(vector float*) from;
  #elif __BIG_ENDIAN__
    __m128 result;
    __m128 temp_ld0 = vec_ld (0, from);
    __m128 temp_ld16 = vec_ld (16, from);
    vector unsigned char permute_selector = vec_lvsl (0, (float *)from);
    result = (__m128) vec_perm (temp_ld0, temp_ld16, permute_selector);
    return result;
  #endif
}

/* Load 4 32-bit floats, unaligned - deprecated - use previous function */
VECLIB_INLINE __m128 vec_load4spu (float const* from)
{
  return vec_loadu4sp (from);
}

/* Load 4 32-bit floats in reverse order, aligned */
VECLIB_INLINE __m128 vec_loadreverse4sp (float const* from)
{
  __m128 result = vec_ld (0, from);
  vector unsigned char permute_vector = {
    0x1C, 0x1D, 0x1E, 0x1F,  0x18, 0x19, 0x1A, 0x1B,  0x14, 0x15, 0x16, 0x17,  0x10, 0x11, 0x12, 0x13
  };
  result = vec_perm (result, result, permute_vector);
  return result;
}

/* Load 32-bit float and splat into 4 32-bit floats */
VECLIB_INLINE __m128 vec_loadsplat4sp (float const* from)
{
  return (__m128) vec_splats (*from);
}

/* Gather 4 32-bit floats */
VECLIB_INLINE __m128 vec_gather4sp (float const* base_addr, __m128i vindex)
{
    float dst[4];
    int idx[4];
    vec_store1q((__m128i *) idx, vindex);
    for (unsigned i = 0; i < 4; i++) {
        dst[i] = base_addr[idx[i]];
    }
    return vec_load4sp(dst);
}

/********************************************************** Set *******************************************************/

/* Set 4 32-bit floats to zero */
VECLIB_INLINE __m128 vec_zero4sp (void)
{
  return (__m128) vec_splats ((float) 0);
}

/* Splat 32-bit float into 4 32-bit floats */
VECLIB_INLINE __m128 vec_splat4sp (float scalar)
{
  #ifdef __ibmxl__
    return (__m128) vec_splats (scalar);
  #elif __GNUC__
    /* LE vec_splats being phased in */
    /* bigger slower alternative */
    __m128_union t;
    t.as_float[0] = scalar;
    t.as_float[1] = scalar;
    t.as_float[2] = scalar;
    t.as_float[3] = scalar;
    return t.as_m128;
  #else
    #error Compiler not supported yet.
  #endif
}

/* Set 4 32-bit floats */
VECLIB_INLINE __m128 vec_set4sp (float f3, float f2, float f1, float f0)
{
  __m128_union t;
  #ifdef __LITTLE_ENDIAN__
    t.as_float[0] = f0;
    t.as_float[1] = f1;
    t.as_float[2] = f2;
    t.as_float[3] = f3;
  #elif __BIG_ENDIAN__
    t.as_float[0] = f3;
    t.as_float[1] = f2;
    t.as_float[2] = f1;
    t.as_float[3] = f0;
  #endif
  return t.as_m128;
}

/* Set 4 32-bit floats reversed */
VECLIB_INLINE __m128 vec_setreverse4sp (float f3, float f2, float f1, float f0)
{
  return (vec_set4sp (f0, f1, f2, f3));
}

/* Set scalar 32-bit float and zero upper 3 32-bit floats */
VECLIB_INLINE __m128 vec_set1sp3z (float scalar)
{
  __m128_union t;
  #ifdef __BIG_ENDIAN__
    t.as_float[3] = scalar;
    t.as_float[2] = 0;
    t.as_float[1] = 0;
    t.as_float[0] = 0;
  #elif __LITTLE_ENDIAN__
    t.as_float[0] = scalar;
    t.as_float[1] = 0;
    t.as_float[2] = 0;
    t.as_float[3] = 0;
  #endif
  return t.as_m128;
}

/* Set scalar 32-bit float and zero upper 3 32-bit floats - deprecated - use previous function */
VECLIB_INLINE __m128 vec_setbot1spscalar4sp (float scalar)
{
  return vec_set1sp3z (scalar);
}

/******************************************************* Store ********************************************************/

/* Store 4 32-bit floats, aligned */
VECLIB_INLINE void vec_store4sp (float* address, __m128 v)
{
  vec_st (v, 0, address);
}

/* Store upper 2 32-bit floats into 2 32-bit floats */
VECLIB_INLINE void vec_storeupper2spof4sp (__m64* to, __m128 from)
{
  __m128_union from_union; from_union.as_m128 = from;
  #ifdef __LITTLE_ENDIAN__
    *to = from_union.as_m64[1];
  #elif __BIG_ENDIAN__
    *to = from_union.as_m64[0];
  #endif
}

/* Store lower 2 32-bit floats into 2 32-bit floats */
VECLIB_INLINE void vec_storelower2spof4sp (__m64* to, __m128 from)
{
  __m128_union from_union; from_union.as_m128 = from;
  #ifdef __LITTLE_ENDIAN__
    *to = from_union.as_m64[0];
  #elif __BIG_ENDIAN__
    *to = from_union.as_m64[1];
  #endif
}

/* Store lower 32-bit float, unaligned */
/* Lower may mean scalar or may mean lower. On Intel those are the same. On PowerPC they are not. */
/* Control which by defining or not defining __LOWER_MEANS_SCALAR_NOT_LOWER__. */
VECLIB_INLINE void vec_storeu4spto1sp (float* address, __m128 v)
{
  __m128_union t;
  t.as_vector_float = v;
  unsigned int element_number;
  #ifdef __LITTLE_ENDIAN__
    #ifdef __LOWER_MEANS_SCALAR_NOT_LOWER__
      element_number = 3;
    #else
      /* Lower means lower not scalar */
      element_number = 0;
    #endif
  #elif __BIG_ENDIAN__
    #ifdef __LOWER_MEANS_SCALAR_NOT_LOWER__
      element_number = 0;
    #else
      /* Lower means lower not scalar */
      element_number = 3;
    #endif
  #endif
  *address = t.as_float[element_number];
}

/* Store lower 32-bit float, unaligned - deprecated - use previous function */
/* Lower may mean scalar or may mean lower. On Intel those are the same. On PowerPC they are not. */
/* Control which by defining or not defining __LOWER_MEANS_SCALAR_NOT_LOWER__. */
VECLIB_INLINE void vec_store4spto1sp (float* address, __m128 v)
{
  vec_storeu4spto1sp (address, v);
}

/* Store 4 32-bit floats in reverse order, aligned */
VECLIB_INLINE void vec_storereverse4sp (float* address, __m128 data)
{
  __m128_union t;
  float temp;
  t.as_m128 = data;

  temp = t.as_float[0];
  t.as_float[0] = t.as_float[3];
  t.as_float[3] = temp;

  temp = t.as_float[1];
  t.as_float[1] = t.as_float[2];
  t.as_float[2] = temp;

  vec_st (t.as_m128, 0, address);
}

/* Store scalar 32-bit float */
/* MS xmmintrin.h */
VECLIB_INLINE float vec_store1spof4sp (__m128 from)
{
  __m128_union from_union; from_union.as_m128 = from;
  float result;
  #ifdef __LITTLE_ENDIAN__
    result = from_union.as_float[0];
  #elif __BIG_ENDIAN__
    result = from_union.as_float[3];
  #endif
  return result;
}

/* Store 4 32-bit floats, unaligned */
VECLIB_INLINE void vec_storeu4sp (float* to, __m128 from)
{
  #if __LITTLE_ENDIAN__
    /* LE Linux ABI requires compilers to handle misaligned pointer dereferences. */
    *(vector float*) to = (vector float) from;
  #elif __BIG_ENDIAN__
    /* Prepare for later generate control mask vector */
    vector signed char all_one = vec_splat_s8( -1 );
    vector signed char all_zero = vec_splat_s8( 0 );
    /* Generate permute vector for the upper part of from */
    vector unsigned char permute_vector = vec_lvsr (0, (unsigned char *) to);
    /* Generate selector vector for the upper part of from */
    vector unsigned char select_vector = vec_perm ((vector unsigned char) all_zero, (vector unsigned char) all_one, permute_vector);

    /* Load from */
    /* Perform a 16-byte load of the original data starting at BoundAlign (to + 0) and BoundAlign (to + 16)*/
    vector unsigned char low = vec_ld (0, (unsigned char *) to);
    vector unsigned char high = vec_ld (16, (unsigned char *) to);
    /* Perform permute, the result will look like:
       original data ... from ... original data */
    vector unsigned char temp_low = vec_perm (low, (vector unsigned char) from, permute_vector);
    low = vec_sel (low, temp_low, select_vector);
    high = vec_perm ((vector unsigned char) from, high, permute_vector);
    /* Store the aligned result for from */
    vec_st (low, 0, (unsigned char *) to);
    vec_st (high, 16, (unsigned char *) to);
  #endif
}

/* Store 4 32-bit floats using a non-temporal memory hint, aligned */
VECLIB_INLINE void vec_store4spstream (float* to, __m128 from)
{
  vec_st (from, 0, to);
  #ifdef __ibmxl__
    /* Non-temporal hint */
    __dcbt ((void *) to);
  #endif
}

/* Store lower 32-bit float splat into 4 32-bit floats, aligned */
VECLIB_INLINE void vec_storesplat1nto4sp (float* to, __m128 from)
{
  vector unsigned char permute_vector = {
    #ifdef __LITTLE_ENDIAN__
      0x10, 0x11, 0x12, 0x13,  0x10, 0x11, 0x12, 0x13,  0x10, 0x11, 0x12, 0x13,  0x10, 0x11, 0x12, 0x13
    #elif __BIG_ENDIAN__
      0x0C, 0x0D, 0x0E, 0x0F,  0x0C, 0x0D, 0x0E, 0x0F,  0x0C, 0x0D, 0x0E, 0x0F,  0x0C, 0x0D, 0x0E, 0x0F
    #endif
  };
  from = (__m128) vec_perm (from, from, permute_vector);
  vec_st (from, 0, to);
}


/********************************************************* Insert *****************************************************/

/* Insert lower 32-bit float into lower of 4 32-bit floats */
VECLIB_INLINE __m128 vec_insert1spintolower4sp (__m128 into, __m128 from)
{
  static const vector unsigned int permute_selector =
    #ifdef __LITTLE_ENDIAN__
      { 0x13121110u, 0x07060504u, 0x0B0A0908u, 0x0F0E0D0Cu };
    #elif __BIG_ENDIAN__
      { 0x00010203u, 0x04050607u, 0x08090A0Bu, 0x1C1D1E1Fu };
    #endif
  return (__m128) vec_perm ((vector float) into, (vector float) from, (vector unsigned char) permute_selector);
}

/* Insert lower 32-bit float into lower of 4 32-bit floats - deprecated - use previous function */
VECLIB_INLINE __m128 vec_insertlowerto4sp (__m128 into, __m128 from)
{
  return vec_insert1spintolower4sp (into, from);
}

/* Extract 32-bit float selected by bits 7:6, insert 32-bit float selected by bits 5:4, */
/* then zero under mask bits 3:0 */
VECLIB_INLINE __m128 vec_insert4sp (__m128 into, __m128 from, const intlit8 control)
{
  int extract_selector = (control >> 6) & 0x3;
  int insert_selector = (control >> 4) & 0x3;
  int zero_selector = control & 0xF;
  static const vector unsigned char extract_selectors[4] = {
    #ifdef __LITTLE_ENDIAN__
      { 0x00, 0x01, 0x02, 0x03,  0x00, 0x01, 0x02, 0x03,  0x00, 0x01, 0x02, 0x03,  0x00, 0x01, 0x02, 0x03 }, /* 1 */
      { 0x04, 0x05, 0x06, 0x07,  0x04, 0x05, 0x06, 0x07,  0x04, 0x05, 0x06, 0x07,  0x04, 0x05, 0x06, 0x07 }, /* 2 */
      { 0x08, 0x09, 0x0A, 0x0B,  0x08, 0x09, 0x0A, 0x0B,  0x08, 0x09, 0x0A, 0x0B,  0x08, 0x09, 0x0A, 0x0B }, /* 3 */
      { 0x0C, 0x0D, 0x0E, 0x0F,  0x0C, 0x0D, 0x0E, 0x0F,  0x0C, 0x0D, 0x0E, 0x0F,  0x0C, 0x0D, 0x0E, 0x0F }, /* 4 */
    #elif __BIG_ENDIAN__
      { 0x0C, 0x0D, 0x0E, 0x0F,  0x0C, 0x0D, 0x0E, 0x0F,  0x0C, 0x0D, 0x0E, 0x0F,  0x0C, 0x0D, 0x0E, 0x0F }, /* 1 */
      { 0x08, 0x09, 0x0A, 0x0B,  0x08, 0x09, 0x0A, 0x0B,  0x08, 0x09, 0x0A, 0x0B,  0x08, 0x09, 0x0A, 0x0B }, /* 2 */
      { 0x04, 0x05, 0x06, 0x07,  0x04, 0x05, 0x06, 0x07,  0x04, 0x05, 0x06, 0x07,  0x04, 0x05, 0x06, 0x07 }, /* 3 */
      { 0x00, 0x01, 0x02, 0x03,  0x00, 0x01, 0x02, 0x03,  0x00, 0x01, 0x02, 0x03,  0x00, 0x01, 0x02, 0x03 }, /* 4 */
    #endif
  };
  vector float extracted = vec_perm (from, from, extract_selectors[extract_selector]);
  static const vector unsigned int insert_selectors[4] = {
    #ifdef __LITTLE_ENDIAN__
      { 0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u }, /* 1 */
      { 0x00000000u, 0xFFFFFFFFu, 0x00000000u, 0x00000000u }, /* 2 */
      { 0x00000000u, 0x00000000u, 0xFFFFFFFFu, 0x00000000u }, /* 3 */
      { 0x00000000u, 0x00000000u, 0x00000000u, 0xFFFFFFFFu }  /* 4 */
    #elif __BIG_ENDIAN__
      { 0x00000000u, 0x00000000u, 0x00000000u, 0xFFFFFFFFu }, /* 1 */
      { 0x00000000u, 0x00000000u, 0xFFFFFFFFu, 0x00000000u }, /* 2 */
      { 0x00000000u, 0xFFFFFFFFu, 0x00000000u, 0x00000000u }, /* 3 */
      { 0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u }, /* 4 */
    #endif
  };
  vector float inserted = vec_sel (into, extracted, insert_selectors[insert_selector]);
  static const vector unsigned int zero_selectors [16] = {
    /* To select left if bit is 0 else right if it is 1 */
    #ifdef __LITTLE_ENDIAN__
      /* Leftmost bit for leftmost element, rightmost bit for rightmost element */
      /* Little endian means the first element below will be rightmost in a VR */
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
      /* Leftmost bit for rightmost element, rightmost bit for leftmost element */
      /* Big endian means the first element below will be leftmost in a VR */
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
  return (__m128) vec_sel (inserted, vec_splats (0.0f), zero_selectors[zero_selector]);
}


/******************************************************** Extract *****************************************************/

/* Extract upper bit of 16 8-bit chars */
VECLIB_INLINE int vec_extractupperbit4sp (__m128 v)
{
  __m128_all_union t;
  t.as_m128 = v;
  int result = 0;
  #ifdef __LITTLE_ENDIAN__
    result |= ((t.as_float[3] < 0) ? 1:0 ) << 3;
    result |= ((t.as_float[2] < 0) ? 1:0 ) << 2;
    result |= ((t.as_float[1] < 0) ? 1:0 ) << 1;
    result |= ((t.as_float[0] < 0) ? 1:0 ) ;
  #elif __BIG_ENDIAN__
    result |= ((t.as_float[0] < 0) ? 1:0 ) << 3;
    result |= ((t.as_float[1] < 0) ? 1:0 ) << 2;
    result |= ((t.as_float[2] < 0) ? 1:0 ) << 1;
    result |= ((t.as_float[3] < 0) ? 1:0 ) ;
  #endif
  return result;
}

/* Extract 32-bit float as an int without conversion */
VECLIB_INLINE int vec_extract4sp (__m128 from, const intlit2 element_number) {
  __m128_union from_union;  from_union.as_m128 = from;
  return from_union.as_hex[element_number & 0x3];
}

/* Extract lower or upper 32+32+32+32-bit floats quad half */
VECLIB_INLINE __m128 vec_extract4spfrom8sp (__m256 from, const intlit1 element_number)
{
  if ((element_number & 1) == 0)
  {
    #ifdef __LITTLE_ENDIAN__
      return from.m128_0;
    #elif __BIG_ENDIAN__
    return from.m128_1;
    #endif
  } else
  {
    #ifdef __LITTLE_ENDIAN__
      return from.m128_1;
    #elif __BIG_ENDIAN__
    return from.m128_0;
    #endif
  }
}


/******************************************** Convert integer to floating-point ***************************************/

/* Convert 2+2 32-bit ints to 32-bit floats */
VECLIB_INLINE __m128 vec_convert2wto4sp (__m64 lower, __m64 upper)
{
  __m64_union lower_union;
  __m64_union upper_union;
  lower_union.as_m64 = lower;
  upper_union.as_m64 = upper;
  lower_union.as_float[0] = (float) lower_union.as_int[0];
  lower_union.as_float[1] = (float) lower_union.as_int[1];
  upper_union.as_float[0] = (float) upper_union.as_int[0];
  upper_union.as_float[1] = (float) upper_union.as_int[1];
  __m128_union result;
    result.as_m64[0] = lower_union.as_m64;
    result.as_m64[1] = upper_union.as_m64;
  return result.as_m128;
}

/* Convert 2+2 32-bit ints to 32-bit floats - deprecated - use previous function */
VECLIB_INLINE __m128 vec_convert22wto4sp (__m64 lower, __m64 upper)
{
  return vec_convert2wto4sp (lower, upper);
}

/* Convert 4 32-bit ints to 4 32-bit floats */
VECLIB_INLINE __m128 vec_convert4swto4sp (__m128i v)
{
  return (__m128) vec_ctf ((vector signed int) v, 0);
}

/* Convert 32-bit int to 32-bit float and insert */
VECLIB_INLINE __m128 vec_convert1swtolower1of4sp (__m128 v, int a)
{  /* check if [0] [3] is right */
  __m128_union result;
  result.as_m128 = v;
  result.as_float[0] = (float) a;
  return result.as_m128;
}

/* Convert 2 32-bit ints to 32-bit floats and insert */
VECLIB_INLINE __m128 vec_convert2swtolower2of4sp (__m128 v , __m64 lower)
{
  __m64_union lower_union;
  lower_union.as_m64 = lower;
  lower_union.as_float[0] = (float) lower_union.as_int[0];
  lower_union.as_float[1] = (float) lower_union.as_int[1];
  __m128_union result;
  result.as_m128 = v;
  result.as_m64[0] = lower_union.as_m64;
  return result.as_m128;
}

/* Convert 64-bit long long to 32-bit float and insert */
VECLIB_INLINE __m128 vec_convert1sdtolower1of4sp (__m128 v , long long int l)
{
  __m128_union result;
  result.as_m128 = v;
  result.as_float[0] = (float) l;
  return result.as_m128;
}

/* Convert lower 4 8-bit chars to 32-bit floats */
VECLIB_INLINE __m128 vec_convertlower4of8sbto4sp(__m64 lower)
{
  __m64_union lower_union;
  lower_union.as_m64 = lower;
  __m128_union result;
  result.as_float[0] = (float) lower_union.as_signed_char[0];
  result.as_float[1] = (float) lower_union.as_signed_char[1];
  result.as_float[2] = (float) lower_union.as_signed_char[2];
  result.as_float[3] = (float) lower_union.as_signed_char[3];
  return result.as_m128;
}

/* Convert lower 4 8-bit unsigned chars to 32-bit floats */
VECLIB_INLINE __m128 vec_convertlower4of8ubto4sp(__m64 lower)
{
  __m64_all_union lower_union;
  lower_union.as_m64 = lower;
  __m128_union result;
    result.as_float[0] = (float) lower_union.as_unsigned_char[0];
    result.as_float[1] = (float) lower_union.as_unsigned_char[1];
    result.as_float[2] = (float) lower_union.as_unsigned_char[2];
    result.as_float[3] = (float) lower_union.as_unsigned_char[3];
  return result.as_m128;
}

/* Convert 4 16-bit shorts to 32-bit floats */
VECLIB_INLINE __m128 vec_convert4shto4sp (__m64 lower)
{
  __m64_union lower_union;
  lower_union.as_m64 = lower;
  __m128_union result;
    result.as_float[0] = (float) lower_union.as_short[0];
    result.as_float[1] = (float) lower_union.as_short[1];
    result.as_float[2] = (float) lower_union.as_short[2];
    result.as_float[3] = (float) lower_union.as_short[3];
  return result.as_m128;
}

/* Convert 4 16-bit unsigned shorts to 32-bit floats */
VECLIB_INLINE __m128 vec_convert4uhto4sp (__m64 lower)
{
  __m64_all_union lower_union;
  lower_union.as_m64 = lower;
  __m128_union result;
    result.as_float[0] = (float) lower_union.as_unsigned_short[0];
    result.as_float[1] = (float) lower_union.as_unsigned_short[1];
    result.as_float[2] = (float) lower_union.as_unsigned_short[2];
    result.as_float[3] = (float) lower_union.as_unsigned_short[3];
  return result.as_m128;
}


/***************************************************** Arithmetic *****************************************************/

/* Horizontally add 2+2 adjacent pairs of 32-bit floats to 4 32-bit floats */
/* Memory left   is { left0,       left1,       left2,         left3 } */
/* Memory right  is { right0,      right1,      right2,        right3 } */
/* Memory result is { left1+left0, left3+left2, right1+right0, right3+right2 } */
VECLIB_INLINE __m128 vec_partialhorizontal2sp (__m128 left, __m128 right)
{
  /* BE register order left   is { left0,       left1,       left2,         left3 } */
  /* BE register order right  is { right0,      right1,      right2,        right3 } */
  /* BE register order result is { left1+left0, left3+left2, right1+right0, right3+right2 } */
  /* LE register order left   is { left3,       left2,       left1,         left0 } */
  /* LE register order right  is { right3,      right2,      right1,        right0 } */
  /* LE register order result is { right3+right2, right1+right0,  left3+left2, left1+left0 } */
  #ifdef __LITTLE_ENDIAN__
    static vector unsigned char addend_1_permute_mask = (vector unsigned char)
      { 0x07,0x06,0x05,0x04, 0x0F,0x0E,0x0D,0x0C, 0x17,0x16,0x15,0x14, 0x1F,0x1E,0x1D,0x1C };
    static vector unsigned char addend_2_permute_mask = (vector unsigned char)
      { 0x03,0x02,0x01,0x00, 0x0B,0x0A,0x09,0x08, 0x1B,0x1A,0x19,0x18, 0x13,0x12,0x11,0x10 };
  #elif __BIG_ENDIAN__
    static vector unsigned char addend_1_permute_mask = (vector unsigned char)
      { 0x04,0x05,0x06,0x07, 0x0C,0x0D,0x0E,0x0F, 0x14,0x15,0x16,0x17, 0x1C,0x1D,0x1E,0x1F };
    static vector unsigned char addend_2_permute_mask = (vector unsigned char)
      { 0x00,0x01,0x02,0x03, 0x08,0x09,0x0A,0x0B, 0x10,0x11,0x12,0x13, 0x18,0x19,0x1A,0x1B };
  #endif
  vector float addend_1 = vec_perm (left, right, addend_1_permute_mask);
  vector float addend_2 = vec_perm (left, right, addend_2_permute_mask);
  return (__m128) vec_add (addend_1, addend_2);
}

/* Horizontally add 2+2 adjacent pairs of 32-bit floats to 4 32-bit floats - deprecated - use previous function */
VECLIB_INLINE __m128 vec_partialhorizontal22sp (__m128 left, __m128 right)
{
  return vec_partialhorizontal2sp (left, right);
}

/* Add 4 32-bit floats */
VECLIB_INLINE __m128 vec_add4sp (__m128 left, __m128 right)
{
  return (__m128) vec_add ((vector float) left, (vector float) right);
}

/* Subtract 4 32-bit floats */
VECLIB_INLINE __m128 vec_subtract4sp (__m128 left, __m128 right)
{
  return (__m128) vec_sub ((vector float) left, (vector float) right);
}

/* Multiply 4 32-bit floats */
VECLIB_INLINE __m128 vec_multiply4sp (__m128 left, __m128 right)
{
  return (__m128) vec_mul ((vector float) left, (vector float) right);
}

/* Divide 4 32-bit floats */
VECLIB_INLINE __m128 vec_divide4sp (__m128 left, __m128 right)
{
  return (__m128) vec_div ((vector float) left, (vector float) right);
}

/* Max 4 32-bit floats */
VECLIB_INLINE __m128 vec_max4sp (__m128 left, __m128 right)
{
  return (__m128) vec_max ((vector float) left, (vector float) right);
}

/* Min 4 32-bit floats */
VECLIB_INLINE __m128 vec_min4sp (__m128 left, __m128 right)
{
  return (__m128) vec_min ((vector float) left, (vector float) right);
}

/**************************************************** Mathematics *****************************************************/

/* Estimate base 2 logs of 4 32-bit floats */
VECLIB_INLINE __m128 vec_log24sp (__m128 v)
/* Note this may be different precision than other implementations. */
{
  return vec_loge (v);
}

/* Dot Product: Multiply 4 32-bit floats (or zeros) under upper 4 bits of mask, sum the 4 products, */
/* then set 4 32-bit floats (or zeros) under lower 4 bits of mask */
#ifdef __ibmxl__
VECLIB_NOINLINE
#else
VECLIB_INLINE
#endif
__m128 vec_dotproduct4sp (__m128 left, __m128 right, const intlit8 multiply_and_result_masks)
{
  #ifdef __ibmxl__
    #pragma option_override (vec_dotproduct4sp, "opt(level,0)")
  #endif
  __m128_all_union result_union;
  static const vector unsigned char all64s =
  { 64,64,64,64, 64,64,64,64, 64,64,64,64, 64,64,64,64 };
  static const vector unsigned char all32s =
  { 32,32,32,32, 32,32,32,32, 32,32,32,32, 32,32,32,32 };
  /* Create masks */
  unsigned int multiply_mask = (multiply_and_result_masks & 0xF0) >> 4;
  unsigned int result_mask   =  multiply_and_result_masks & 0x0F;
  __m128_all_union multiply_element_mask;
  __m128_all_union result_element_mask;
  multiply_element_mask.as_vector_bool_int = (vector bool int) expand_bit_to_word_masks[multiply_mask];
  result_element_mask.as_vector_unsigned_int = (vector unsigned int) permute_highest_word_to_words_masks[result_mask];
  /* Calculate products */
  __m128_all_union masked_left;
  masked_left.as_m128  = (__m128) vec_and (left, multiply_element_mask.as_vector_bool_int);
  __m128_all_union masked_right;
  masked_right.as_m128 = (__m128) vec_and (right, multiply_element_mask.as_vector_bool_int);
  __m128_all_union products;
  products.as_m128 = (__m128) vec_madd (masked_left .as_vector_float,
                                        masked_right.as_vector_float,
                                        vec_splats (0.f));
  /* Horizontally add products into highest element, garbage in rest */
  __m128_all_union t;
  #ifdef USE_VEC_SLD
    t.as_m128 = (__m128) vec_add (products.as_vector_float,
                                  vec_sld (products.as_vector_float, products.as_vector_float, 64/8));
  #else
    t.as_m128 = (__m128) vec_add (products.as_vector_float,
                                  vec_slo (products.as_vector_float, all64s));
  #endif
  __m128_all_union s;
  #ifdef USE_VEC_SLD
    s.as_m128 = (__m128) vec_add (t.as_vector_float,
                                  vec_sld (t.as_vector_float, t.as_vector_float, 32/8));
  #else
    s.as_m128 = (__m128) vec_add (t.as_vector_float,
                                  vec_slo (t.as_vector_float, all32s));
  #endif
  /* Permute highest element to result element(s) */
  result_union.as_vector_float = vec_perm (vec_splats (0.f), s.as_vector_float,
                                                             result_element_mask.as_vector_unsigned_char);
  return result_union.as_m128;
}

/* Add scalar 32-bit floats keeping upper left */
VECLIB_INLINE __m128 vec_add1spof4sp (__m128 left, __m128 right)
{
  vector float all_zero = vec_splats ((float) 0.0);
  vector unsigned int select_vector = {
    0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
  };
  vector float extracted_right = vec_sel (all_zero, (vector float) right, select_vector);
  vector float result = vec_add (extracted_right, (vector float) left); /* handle -0 + 0 */
  return (__m128) result;
}

/* Square roots of 4 32-bit floats */
VECLIB_INLINE __m128 vec_squareroot4sp (__m128 v)
{
  return (__m128) vec_sqrt (v);
}

/* Subtract scalar 32-bit floats keeping upper left */
VECLIB_INLINE __m128 vec_subtract1spof4sp (__m128 left, __m128 right)
{
  vector float all_zero = vec_splats ((float) 0.0);
  vector unsigned int select_vector = {
    0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
  };
  vector float extracted_right = vec_sel (all_zero, (vector float) right, select_vector);
  vector float result = vec_sub ((vector float) left, extracted_right); /* handle -0 + 0 */
  return (__m128) result;
}


/* Multiply scalar 32-bit floats keeping upper left */
VECLIB_INLINE __m128 vec_multiply1spof4sp (__m128 left, __m128 right)
{
  vector float all_ones = vec_splats ((float) 1.0);
  vector unsigned int select_vector = {
    0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
  };
  vector float extracted_right = vec_sel (all_ones, (vector float) right, select_vector);
  vector float result = vec_mul ((vector float) left, extracted_right);
  return (__m128) result;
}

/* Divide scalar 32-bit floats keeping upper left */
#ifdef VECLIB_VSX
VECLIB_INLINE __m128 vec_divide1spof4sp (__m128 left, __m128 right)
{
  vector float all_ones = vec_splats ((float) 1.0);
  vector unsigned int select_vector = {
      0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
  };
  vector float extracted_right = vec_sel (all_ones, (vector float) right, select_vector);
  vector float result = vec_div ((vector float) left, extracted_right);
  return (__m128) result;
}
#endif

/* Square root of scalar 32-bit float keeping upper elements */
/* Note square root may be less accurate than other implementations */
VECLIB_INLINE __m128 vec_squareroot1spof4sp (__m128 v)
{
  vector float all_zero = vec_splats ((float) 0.0);
  vector unsigned int select_vector = {
    0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
  };
  vector float first_res = vec_sqrt ((vector float) v);
  vector bool int is_zero_mask = vec_cmpeq (all_zero, (vector float) v);
  vector float inter_res = (__m128) vec_sel (first_res, all_zero, is_zero_mask);
  vector float result = vec_sel ((vector float) v, inter_res, select_vector);
  return (__m128) result;
}

/* Max scalar 32-bit floats keeping upper left */
VECLIB_INLINE __m128 vec_max1spof4sp (__m128 left, __m128 right)
{
  vector unsigned int select_vector = {
    0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
  };
  vector float inter_res = vec_max ((vector float) left, (vector float) right);
  vector float result = vec_sel ((vector float) left, inter_res, select_vector);
  return (__m128) result;
}

/* Min scalar 32-bit floats keeping upper left */
VECLIB_INLINE __m128 vec_min1spof4sp (__m128 left, __m128 right)
{
  vector unsigned int select_vector = {
      0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
  };
  vector float inter_res = vec_min ((vector float) left, (vector float) right);
  vector float result = vec_sel ((vector float) left, inter_res, select_vector);
  return (__m128) result;
}

/* Compute approximate (<1.5*2*-12) reciprocals of 4 floats */
/* Note more accurate than other implementations */
VECLIB_INLINE __m128 vec_reciprocalestimate4sp (__m128 v)
{
  return (__m128) vec_re ((vector float) v);
}

/* Compute approximate (<1.5*2*-12) reciprocal of scalar float keeping upper elements */
/* Note more accurate than other implementations */
VECLIB_INLINE __m128 vec_reciprocalestimate1spof4sp (__m128 v)
{
  vector unsigned int select_vector = {
    0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
  };
  vector float inter_res = vec_re ((vector float) v);
  vector float result = vec_sel ((vector float) v, inter_res, select_vector);
  return (__m128) result;
}

/* Compute approximate (<1.5*2*-12) reciprocal square roots of 4 floats */
/* Note more accurate than other implementations */
VECLIB_INLINE __m128 vec_reciprocalsquarerootestimate4sp (__m128 v)
{
  return (__m128) vec_rsqrte ((vector float) v);
}


/* Compute approximate (<1.5*2*-12) reciprocal square root of scalar float keeping upper elements*/
/* Note more accurate than other implementations */
VECLIB_INLINE __m128 vec_reciprocalsquarerootestimate1spof4sp (__m128 v)
{
  /* Need to keep the negative on the other elements */
  vector float all_zero = vec_splats ((float) 1.0);
  vector unsigned int select_vector = {
    0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
  };
  vector float extracted_right = vec_sel (all_zero, (vector float) v, select_vector);
  vector float inter_res = vec_rsqrte ((vector float) extracted_right);
  vector float result = vec_sel ((vector float) v, inter_res, select_vector);
  return (__m128) result;
}

/* Floor 4 32-bit floats */
VECLIB_INLINE __m128 vec_Floor4sp (__m128 v) {
  return vec_floor (v);
}

/******************************************************** Boolean *****************************************************/

/* Bitwise 128-bit or */
VECLIB_INLINE __m128 vec_bitwiseor4sp (__m128 left, __m128 right)
{
  return (__m128) vec_or (left, right);
}

/* Bitwise 128-bit and */
VECLIB_INLINE __m128 vec_bitwiseand4sp (__m128 left, __m128 right)
{
  return (__m128) vec_and (left, right);
}

/* Bitwise 128-bit and - deprecated - use previous function */
VECLIB_INLINE __m128 vec_bitand4sp (__m128 left, __m128 right)
{
  return vec_bitwiseand4sp (left, right);
}

/* Bitwise 128-bit and not (reversed) */
VECLIB_INLINE __m128 vec_bitwiseandnotleft4sp (__m128 left, __m128 right)
{
  return (__m128) vec_andc (right, left);
}

/* Bitwise 128-bit and not (reversed) - deprecated - use previous function */
VECLIB_INLINE __m128 vec_bitandnotleft4sp (__m128 left, __m128 right)
{
  return vec_bitwiseandnotleft4sp (left, right);
}

/* Bitwise 128-bit exclusive or */
VECLIB_INLINE __m128 vec_bitwisexor4sp (__m128 left, __m128 right)
{
  return (__m128) vec_xor ((vector float) left, (vector float) right);
}

/******************************************************* Permute ******************************************************/

/* Shuffle 4+4 32-bit floats into 4 32-bit floats using mask */
/* Result 127:96 = left[mask[7:6]], 95:64 = left[mask[5:4]], 63:32 = right[mask[3:2]], 31:0 = right[mask[1:0]]. */
/* Elements count 0 - N-1 from right to left. */
VECLIB_INLINE __m128 vec_shufflepermute4sp (__m128 left, __m128 right, unsigned int element_selectors)
{
  unsigned long element_selector_10 =  element_selectors       & 0x03;
  unsigned long element_selector_32 = (element_selectors >> 2) & 0x03;
  unsigned long element_selector_54 = (element_selectors >> 4) & 0x03;
  unsigned long element_selector_76 = (element_selectors >> 6) & 0x03;
  #ifdef __LITTLE_ENDIAN__
    const static unsigned int permute_selectors_from_left_operand  [4] = { 0x03020100u, 0x07060504u, 0x0B0A0908u, 0x0F0E0D0Cu };
    const static unsigned int permute_selectors_from_right_operand [4] = { 0x13121110u, 0x17161514u, 0x1B1A1918u, 0x1F1E1D1Cu };
  #elif __BIG_ENDIAN__
    const static unsigned int permute_selectors_from_left_operand  [4] = { 0x00010203u, 0x04050607u, 0x08090A0Bu, 0x0C0D0E0Fu };
    const static unsigned int permute_selectors_from_right_operand [4] = { 0x10111213u, 0x14151617u, 0x18191A1Bu, 0x1C1D1E1Fu };
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
  return (vector float) vec_perm ((vector unsigned char) left, (vector unsigned char) right,
                                  permute_selectors.as_vector_unsigned_char);
}

/* Shuffle 4+4 32-bit floats into 4 32-bit floats using mask - deprecated - use previous function */
/* Result 127:96 = left[mask[7:6]], 95:64 = left[mask[5:4]], 63:32 = right[mask[3:2]], 31:0 = right[mask[1:0]]. */
/* Elements count 0 - N-1 from right to left. */
VECLIB_INLINE __m128 vec_shufflepermute44sp (__m128 left, __m128 right, unsigned int element_selectors)
{
  return vec_shufflepermute4sp (left, right, element_selectors);
}

/* Blend 4+4 32-bit floats under mask to 4 32-bit floats */
VECLIB_INLINE __m128 vec_blendpermute4sp (__m128 left, __m128 right, const intlit4 mask)
{
  static const vector unsigned char permute_selector[16] = {
    /* To select left element for 0 or right element for 1 */
    #ifdef __LITTLE_ENDIAN__
      { 0x0F,0x0E,0x0D,0x0C, 0x0B,0x0A,0x09,0x08, 0x07,0x06,0x05,0x04, 0x03,0x02,0x01,0x00 }, /* 0000 */
      { 0x1F,0x1E,0x1D,0x1C, 0x0B,0x0A,0x09,0x08, 0x07,0x06,0x05,0x04, 0x03,0x02,0x01,0x00 }, /* 0001 */
      { 0x0F,0x0E,0x0D,0x0C, 0x1B,0x1A,0x19,0x18, 0x07,0x06,0x05,0x04, 0x03,0x02,0x01,0x00 }, /* 0010 */
      { 0x1F,0x1E,0x1D,0x1C, 0x1B,0x1A,0x19,0x18, 0x07,0x06,0x05,0x04, 0x03,0x02,0x01,0x00 }, /* 0011 */
      { 0x0F,0x0E,0x0D,0x0C, 0x0B,0x0A,0x09,0x08, 0x17,0x16,0x15,0x14, 0x03,0x02,0x01,0x00 }, /* 0100 */
      { 0x1F,0x1E,0x1D,0x1C, 0x0B,0x0A,0x09,0x08, 0x17,0x16,0x15,0x14, 0x03,0x02,0x01,0x00 }, /* 0101 */
      { 0x0F,0x0E,0x0D,0x0C, 0x1B,0x1A,0x19,0x18, 0x17,0x16,0x15,0x14, 0x03,0x02,0x01,0x00 }, /* 0110 */
      { 0x1F,0x1E,0x1D,0x1C, 0x1B,0x1A,0x19,0x18, 0x17,0x16,0x15,0x14, 0x03,0x02,0x01,0x00 }, /* 0111 */
      { 0x0F,0x0E,0x0D,0x0C, 0x0B,0x0A,0x09,0x08, 0x07,0x06,0x05,0x04, 0x13,0x02,0x01,0x00 }, /* 1000 */
      { 0x1F,0x1E,0x1D,0x1C, 0x0B,0x0A,0x09,0x08, 0x07,0x06,0x05,0x04, 0x13,0x02,0x01,0x10 }, /* 1001 */
      { 0x0F,0x0E,0x0D,0x0C, 0x1B,0x1A,0x19,0x18, 0x07,0x06,0x05,0x04, 0x13,0x02,0x11,0x00 }, /* 1010 */
      { 0x1F,0x1E,0x1D,0x1C, 0x1B,0x1A,0x19,0x18, 0x07,0x06,0x05,0x04, 0x13,0x02,0x11,0x01 }, /* 1011 */
      { 0x0F,0x0E,0x0D,0x0C, 0x0B,0x0A,0x09,0x08, 0x17,0x16,0x15,0x14, 0x13,0x12,0x01,0x00 }, /* 1100 */
      { 0x1F,0x1E,0x1D,0x1C, 0x0B,0x0A,0x09,0x08, 0x17,0x16,0x15,0x14, 0x13,0x12,0x01,0x10 }, /* 1101 */
      { 0x0F,0x0E,0x0D,0x0C, 0x1B,0x1A,0x19,0x18, 0x17,0x16,0x15,0x14, 0x13,0x12,0x11,0x00 }, /* 1110 */
      { 0x1F,0x1E,0x1D,0x1C, 0x1B,0x1A,0x19,0x18, 0x17,0x16,0x15,0x14, 0x03,0x02,0x01,0x00 }  /* 1111 */
    #elif __BIG_ENDIAN__
      { 0x00,0x01,0x02,0x03, 0x04,0x05,0x06,0x07, 0x08,0x09,0x0A,0x0B, 0x0C,0x0D,0x0E,0x0F }, /* 0000 */
      { 0x00,0x01,0x02,0x03, 0x04,0x05,0x06,0x07, 0x08,0x09,0x0A,0x0B, 0x1C,0x1D,0x1E,0x1F }, /* 0001 */
      { 0x00,0x01,0x02,0x03, 0x04,0x05,0x06,0x07, 0x18,0x19,0x1A,0x1B, 0x0C,0x0D,0x0E,0x0F }, /* 0010 */
      { 0x00,0x01,0x02,0x03, 0x04,0x05,0x06,0x07, 0x18,0x19,0x1A,0x1B, 0x1C,0x1D,0x1E,0x1F }, /* 0011 */
      { 0x00,0x01,0x02,0x03, 0x14,0x15,0x16,0x17, 0x08,0x09,0x0A,0x0B, 0x0C,0x0D,0x0E,0x0F }, /* 0100 */
      { 0x00,0x01,0x02,0x03, 0x14,0x15,0x16,0x17, 0x08,0x09,0x0A,0x0B, 0x1C,0x1D,0x1E,0x1F }, /* 0101 */
      { 0x00,0x01,0x02,0x03, 0x14,0x15,0x16,0x17, 0x18,0x19,0x1A,0x1B, 0x0C,0x0D,0x0E,0x0F }, /* 0110 */
      { 0x00,0x01,0x02,0x03, 0x14,0x15,0x16,0x17, 0x18,0x19,0x1A,0x1B, 0x1C,0x1D,0x1E,0x1F }, /* 0111 */
      { 0x10,0x11,0x12,0x13, 0x04,0x05,0x06,0x07, 0x08,0x09,0x0A,0x0B, 0x0C,0x0D,0x0E,0x0F }, /* 1000 */
      { 0x10,0x11,0x12,0x13, 0x04,0x05,0x06,0x07, 0x08,0x09,0x0A,0x0B, 0x1C,0x1D,0x1E,0x1F }, /* 1001 */
      { 0x10,0x11,0x12,0x13, 0x04,0x05,0x06,0x07, 0x18,0x19,0x1A,0x1B, 0x0C,0x0D,0x0E,0x0F }, /* 1010 */
      { 0x10,0x11,0x12,0x13, 0x04,0x05,0x06,0x07, 0x18,0x19,0x1A,0x1B, 0x1C,0x1D,0x1E,0x1F }, /* 1011 */
      { 0x10,0x11,0x12,0x13, 0x14,0x15,0x16,0x17, 0x08,0x09,0x0A,0x0B, 0x0C,0x0D,0x0E,0x0F }, /* 1100 */
      { 0x10,0x11,0x12,0x13, 0x14,0x15,0x16,0x17, 0x08,0x09,0x0A,0x0B, 0x1C,0x1D,0x1E,0x1F }, /* 1101 */
      { 0x10,0x11,0x12,0x13, 0x14,0x15,0x16,0x17, 0x18,0x19,0x1A,0x1B, 0x0C,0x0D,0x0E,0x0F }, /* 1110 */
      { 0x10,0x11,0x12,0x13, 0x14,0x15,0x16,0x17, 0x18,0x19,0x1A,0x1B, 0x1C,0x1D,0x1E,0x1F }  /* 1111 */
    #endif
  };
  return (__m128) vec_perm ((vector float) left, (vector float) right, permute_selector[mask & 0xF]);
}

/* Blend 4+4 32-bit floats under mask to 4 32-bit floats - deprecated - use previous function */
VECLIB_INLINE __m128 vec_blendpermute44sp (__m128 left, __m128 right, const intlit4 mask)
{
  return vec_blendpermute4sp (left, right, mask);
}

/* Blend 4+4 32-bit floats under mask to 4 32-bit floats */
VECLIB_INLINE __m128 vec_permutevr4sp (__m128 left, __m128 right, __m128 mask)
{
  /* Upper bit of each element mask selects 0 = left 1 = right */
  vector bool int select_mask = vec_cmplt ((vector signed int) mask, vec_splats (0));  /* convert upper bits to zeros or ones mask */
  return (__m128) vec_sel (left, right, select_mask);
}

/* Blend 4+4 32-bit floats under mask to 4 32-bit floats - deprecated - use previous function */
VECLIB_INLINE __m128 vec_permutevr44sp (__m128 left, __m128 right, __m128 mask)
{
  return vec_permutevr4sp (left, right, mask);
}

/* Extract upper 2 32-bit floats and insert into lower 2 of 4 32-bit floats */
VECLIB_INLINE __m128 vec_extractupper2spinsertlower2spof4sp (__m128 upper_to_upper, __m128 upper_to_lower)
{
  vector unsigned char permute_selector = {
    #ifdef __LITTLE_ENDIAN__
      0x18, 0x19, 0x1A, 0x1B,  0x1C, 0x1D, 0x1E, 0x1F,  0x08, 0x09, 0xA, 0xB,    0x0C, 0x0D, 0x0E, 0x0F
    #elif __BIG_ENDIAN__
      0x00, 0x01, 0x02, 0x03,  0x04, 0x05, 0x06, 0x07,  0x10, 0x11, 0x12, 0x13,  0x14, 0x15, 0x16, 0x17
    #endif
  };
  return vec_perm ((vector float) upper_to_upper, (vector float) upper_to_lower, permute_selector);
}

/* Splat second 32-bit float to 4 32-bit floats - (A0, A1, A2, A3) ==> (A1, A1, A3, A3) */
VECLIB_INLINE __m128  vec_extractoddsptoevensp (__m128 a) {
  vector unsigned char permute_selector = {
    #ifdef __LITTLE_ENDIAN__
      0x04, 0x05, 0x6, 0x7,  0x04, 0x05, 0x6, 0x7,   0x0C, 0x0D, 0x0E, 0x0F,   0x0C, 0x0D, 0x0E, 0x0F
    #elif __BIG_ENDIAN__
      0x00, 0x01, 0x2, 0x3,   0x00, 0x01, 0x2, 0x3,   0x08, 0x09, 0x0A, 0x0B,   0x08, 0x09, 0x0A, 0x0B
    #endif
  };
  return vec_perm(a, a, permute_selector);
}

/* Splat low 32-bit float to 4 32-bit floats - (A0, A1, A2, A3) ==> (A0, A0, A2, A2) */
VECLIB_INLINE __m128 vec_extractevensptooddsp (__m128 a) {
  vector unsigned char permute_selector = {
    #ifdef __LITTLE_ENDIAN__
      0x00, 0x01, 0x2, 0x3,   0x00, 0x01, 0x2, 0x3,   0x08, 0x09, 0x0A, 0x0B,   0x08, 0x09, 0x0A, 0x0B
    #elif __BIG_ENDIAN__
      0x04, 0x05, 0x6, 0x7,  0x04, 0x05, 0x6, 0x7,   0x0C, 0x0D, 0x0E, 0x0F,   0x0C, 0x0D, 0x0E, 0x0F
    #endif
  };
  return vec_perm(a, a, permute_selector);
}

/* Extract lower 2 32-bit floats and insert into upper 2 of 4 32-bit floats */
VECLIB_INLINE __m128 vec_extractlower2spinsertupper2spof4sp (__m128 lower_to_lower, __m128 lower_to_upper)
{
  vector unsigned char permute_selector = {
    #ifdef __LITTLE_ENDIAN__
      0x10, 0x11, 0x12, 0x13,  0x14, 0x15, 0x16, 0x17,  0x00, 0x01, 0x02, 0x03,  0x04, 0x05, 0x06, 0x07
    #elif __BIG_ENDIAN__
      0x08, 0x09, 0x0A, 0x0B,  0x0C, 0x0D, 0x0E, 0x0F,  0x18, 0x19, 0x1A, 0x1B,  0x1C, 0x1D, 0x1E, 0x1F
    #endif
  };
  return vec_perm ((vector float) lower_to_upper, (vector float) lower_to_lower, permute_selector);
}

/******************************************************* Compare ******************************************************/

/* forward declaration */
VECLIB_INLINE __m128 vec_comparenotnans4sp (__m128 left, __m128 right);

/* Compare eq / neq */

/* Compare 4 32-bit floats for == to mask */
VECLIB_INLINE __m128 vec_compareeq4sp (__m128 left, __m128 right)
{
  return (__m128) vec_cmpeq ((vector float) left, (vector float) right);
}

/* Compare 4 32-bit floats for == to mask - deprecated - use previous function */
VECLIB_INLINE __m128 vec_compareeq_4sp (__m128 left, __m128 right)
{
  return vec_compareeq4sp (right, left);
}

/* Compare 4 32-bit floats for != to mask */
VECLIB_INLINE __m128 vec_comparene4sp (__m128 left, __m128 right)
{
  __m128_union leftx;
  leftx.as_m128 = vec_compareeq4sp (left, right);
  __m128_union rightx;
  rightx.as_m128 = (__m128) vec_splats ((unsigned char) 0xFF);
  return (__m128) vec_xor ((vector float) leftx.as_m128, (vector float) rightx.as_m128);
}

/* Compare scalar 32-bit floats for == to mask keeping upper elements */
VECLIB_INLINE __m128 vec_compareeq1of4sp (__m128 left, __m128 right)
{
    vector unsigned int select_vector = {
        0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
    };
    vector float inter_res = vec_compareeq4sp ((vector float) left, (vector float) right);
    vector float result = vec_sel ((vector float) left, inter_res, select_vector);
    return (__m128) result;
}

/* Compare scalar 32-bit floats for !=  to mask keeping upper elements */
VECLIB_INLINE __m128 vec_comparene1of4sp (__m128 left, __m128 right)
{
    vector unsigned int select_vector = {
        0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
    };
    vector float inter_res = vec_comparene4sp ((vector float) left, (vector float) right);
    vector float result = vec_sel ((vector float) left, inter_res, select_vector);
    return (__m128) result;
}

/* Compare scalar 32-bit floats for == to bool */
VECLIB_INLINE int vec_compareeq1of4sptobool (__m128 left, __m128 right)
{
  __m128_union res_union;
  __m128_union nan_union;
  res_union.as_m128 = vec_compareeq4sp(left, right);
  nan_union.as_m128 = vec_comparenotnans4sp(left, right);
  /* Return 1s for NaN */
  /* All these should do boolean ands/ors as vectors then a single convert to int avoiding short circuit operations */
  return nan_union.as_hex_unsigned[0] == 0x00000000u || res_union.as_hex_unsigned[0] == 0xFFFFFFFFu;
}

/* Compare scalar 32-bit floats for != to bool */
VECLIB_INLINE int vec_comparene1of4sptobool (__m128 left, __m128 right)
{
  __m128_union res_union;
  __m128_union nan_union;
  res_union.as_m128 = vec_comparene4sp(left, right);
  nan_union.as_m128 = vec_comparenotnans4sp(left, right);
  /* Return 0s for NaN */
  return nan_union.as_hex_unsigned[0] == 0xFFFFFFFFu && res_union.as_hex_unsigned[0] == 0xFFFFFFFFu;
}

/* Compare scalar 32-bit floats for == to bool QNaNs nonsignaling */
VECLIB_INLINE int vec_compareeq1of4sptoboolnonsignaling (__m128 left, __m128 right)
{
  __m128_union res_union;
  __m128_union nan_union;
  res_union.as_m128 = vec_compareeq4sp(left, right);
  nan_union.as_m128 = vec_comparenotnans4sp(left, right);
  /* Return 1s for NaN */
  return nan_union.as_hex_unsigned[0] == 0x00000000u || res_union.as_hex_unsigned[0] == 0xFFFFFFFFu;
}

/* Compare scalar 32-bit floats for != to bool, QNaNs nonsignaling */
VECLIB_INLINE int vec_comparene1of4sptoboolnonsignaling (__m128 left, __m128 right)
{
  __m128_union res_union;
  __m128_union nan_union;
  res_union.as_m128 = vec_comparene4sp(left, right);
  nan_union.as_m128 = vec_comparenotnans4sp(left, right);
  /* Return 0s for NaN */
  return nan_union.as_hex_unsigned[0] == 0xFFFFFFFFu && res_union.as_hex_unsigned[0] == 0xFFFFFFFFu;
}

/* Compare lt / nlt */

/* Compare 4 32-bit float for < to vector mask */
VECLIB_INLINE __m128 vec_comparelt4sp (__m128 left, __m128 right)
{
  return (__m128) vec_cmplt ((vector float) left, (vector float) right);
}

/* Compare 4 32-bit float for < to vector mask - deprecated - use previous function */
VECLIB_INLINE __m128 vec_comparelt_4sp (__m128 left, __m128 right)
{
  return vec_comparelt4sp (left, right);
}

/* Compare 4 32-bit floats for !< to mask */
VECLIB_INLINE __m128 vec_comparenotlt4sp (__m128 left, __m128 right)
{
  __m128_union leftx;
  leftx.as_m128 = vec_comparelt4sp (left, right); /* vec_comparele_4sp deprecated */
  __m128_union rightx;
  rightx.as_m128 = (__m128) vec_splats ((unsigned char) 0xFF);

  return (__m128) vec_xor ((vector float) leftx.as_m128, (vector float) rightx.as_m128);
}

/* Compare scalar 32-bit floats for < to mask keeping upper elements */
VECLIB_INLINE __m128 vec_comparelt1of4sp (__m128 left, __m128 right)
{
    vector unsigned int select_vector = {
        0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
    };
    vector float inter_res = vec_comparelt4sp ((vector float) left, (vector float) right);
    vector float result = vec_sel ((vector float) left, inter_res, select_vector);
    return (__m128) result;
}

/* Compare scalar 32-bit floats for !< to mask keeping upper elements */
VECLIB_INLINE __m128 vec_comparenotlt1of4sp (__m128 left, __m128 right)
{
    vector unsigned int select_vector = {
        0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
    };
    vector float inter_res = vec_comparenotlt4sp ((vector float) left, (vector float) right);
    vector float result = vec_sel ((vector float) left, inter_res, select_vector);
    return (__m128) result;
}

/* Compare scalar 32-bit floats for < to bool */
VECLIB_INLINE int vec_comparelt1of4sptobool (__m128 left, __m128 right)
{
  __m128_union res_union;
  __m128_union nan_union;
  res_union.as_m128 = vec_comparelt4sp(left, right);
  nan_union.as_m128 = vec_comparenotnans4sp(left, right);
  /* Return 1s for NaN */
  return nan_union.as_hex_unsigned[0] == 0x00000000u || res_union.as_hex_unsigned[0] == 0xFFFFFFFFu;
}

/* Compare scalar 32-bit floats for < to bool QNaNs nonsignaling */
VECLIB_INLINE int vec_comparelt1of4sptoboolnonsignaling (__m128 left, __m128 right)
{
  __m128_union res_union;
  __m128_union nan_union;
  res_union.as_m128 = vec_comparelt4sp(left, right);
  nan_union.as_m128 = vec_comparenotnans4sp(left, right);
  /* Return 1s for NaN */
  return nan_union.as_hex_unsigned[0] == 0x00000000u || res_union.as_hex_unsigned[0] == 0xFFFFFFFFu;
}

/* Compare le / nle */

/* Compare 4 32-bit float for <= to vector mask */
VECLIB_INLINE __m128 vec_comparele4sp (__m128 left, __m128 right)
{
  return (__m128) vec_cmple ((vector float) left, (vector float) right);
}

/* Compare 4 32-bit float for <= to vector mask - deprecated - use previous function */
VECLIB_INLINE __m128 vec_comparele_4sp (__m128 left, __m128 right)
{
  return vec_comparele4sp (left, right);
}

/* Compare 4 32-bit floats for !<= to mask */
VECLIB_INLINE __m128 vec_comparenotle4sp (__m128 left, __m128 right)
{
  __m128_union leftx;
  leftx.as_m128 = vec_comparele4sp (left, right); /* vec_comparele_4sp deprecated */
  __m128_union rightx;
  rightx.as_m128 = (__m128) vec_splats ((unsigned char) 0xFF);
  return (__m128) vec_xor ((vector float) leftx.as_m128, (vector float) rightx.as_m128);
}

/* Compare scalar 32-bit floats for <= to mask keeping upper elements */
VECLIB_INLINE __m128 vec_comparele1of4sp (__m128 left, __m128 right)
{
  vector unsigned int select_vector = {
    0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
  };
  vector float inter_res = vec_comparele4sp ((vector float) left, (vector float) right);
  vector float result = vec_sel ((vector float) left, inter_res, select_vector);
  return (__m128) result;
}

/* Compare scalar 32-bit floats for !<= to mask keeping upper elements */
VECLIB_INLINE __m128 vec_comparenotle1of4sp (__m128 left, __m128 right)
{
  vector unsigned int select_vector = {
    0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
  };
  vector float inter_res = vec_comparenotle4sp ((vector float) left, (vector float) right);
  vector float result = vec_sel ((vector float) left, inter_res, select_vector);
  return (__m128) result;
}

/* Compare scalar 32-bit floats for <= to bool */
VECLIB_INLINE int vec_comparele1of4sptobool (__m128 left, __m128 right)
{
  __m128_union res_union;
  __m128_union nan_union;
  res_union.as_m128 = vec_comparele4sp(left, right);
  nan_union.as_m128 = vec_comparenotnans4sp(left, right);
  /* Return 1s for NaN */
  return nan_union.as_hex_unsigned[0] == 0x00000000u || res_union.as_hex_unsigned[0] == 0xFFFFFFFFu;
}

/* Compare scalar 32-bit floats for <= to bool QNaNs nonsignaling */
VECLIB_INLINE int vec_comparele1of4sptoboolnonsignaling (__m128 left, __m128 right)
{
  __m128_union res_union;
  __m128_union nan_union;
  res_union.as_m128 = vec_comparele4sp(left, right);
  nan_union.as_m128 = vec_comparenotnans4sp(left, right);
  /* Return 1s for NaN */
  return nan_union.as_hex_unsigned[0] == 0x00000000u || res_union.as_hex_unsigned[0] == 0xFFFFFFFFu;
}

/* Compare gt / ngt */

/* Compare 4 32-bit float for > to vector mask */
VECLIB_INLINE __m128 vec_comparegt4sp (__m128 left, __m128 right)
{
  return (__m128) vec_cmpgt ((vector float) left, (vector float) right);
}

/* Compare 4 32-bit float for > to vector mask - deprecated - use previous function */
VECLIB_INLINE __m128 vec_comparegt_4sp (__m128 left, __m128 right)
{
  return vec_comparegt4sp (left, right);
}

/* Compare 4 32-bit floats for !> to mask */
VECLIB_INLINE __m128 vec_comparenotgt4sp (__m128 left, __m128 right)
{
  __m128_union leftx;
  leftx.as_m128 = vec_comparegt4sp (left, right);
  __m128_union rightx;
  rightx.as_m128 = (__m128) vec_splats ((unsigned char) 0xFF);

  return (__m128) vec_xor ((vector float) leftx.as_m128, (vector float) rightx.as_m128);
}

/* Compare scalar 32-bit floats for > to mask keeping upper elements */
VECLIB_INLINE __m128 vec_comparegtlower1of4sp (__m128 left, __m128 right)
{
    vector unsigned int select_vector = {
        0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
    };
    vector float inter_res = vec_comparegt4sp ((vector float) left, (vector float) right);
    vector float result = vec_sel ((vector float) left, inter_res, select_vector);
    return (__m128) result;
}

/* Compare scalar 32-bit floats for !> to mask keeping upper elements */
VECLIB_INLINE __m128 vec_comparenotgtlower1of4sp (__m128 left, __m128 right)
{
    vector unsigned int select_vector = {
        0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
    };
    vector float inter_res = vec_comparenotgt4sp ((vector float) left, (vector float) right);
    vector float result = vec_sel ((vector float) left, inter_res, select_vector);
    return (__m128) result;
}

/* Compare scalar 32-bit floats for > to bool */
VECLIB_INLINE int vec_comparegt1of4sptobool (__m128 left, __m128 right)
{
  __m128_union res_union;
  __m128_union nan_union;
  res_union.as_m128 = vec_comparegt4sp(left, right);
  nan_union.as_m128 = vec_comparenotnans4sp(left, right);
  /* Return 0s for NaN */
  return nan_union.as_hex_unsigned[0] == 0xFFFFFFFFu && res_union.as_hex_unsigned[0] == 0xFFFFFFFFu;
}

/* Compare scalar 32-bit floats for > to bool QNaNs nonsignaling */
VECLIB_INLINE int vec_comparegt1of4sptoboolnonsignaling (__m128 left, __m128 right)
{
  __m128_union res_union;
  __m128_union nan_union;
  res_union.as_m128 = vec_comparegt4sp(left, right);
  nan_union.as_m128 = vec_comparenotnans4sp(left, right);
  /* Return 0s for NaN */
  return nan_union.as_hex_unsigned[0] == 0xFFFFFFFFu && res_union.as_hex_unsigned[0] == 0xFFFFFFFFu;
}

/* Compare ge / nge */

/* Compare 4 32-bit floats for >= to mask */
VECLIB_INLINE __m128 vec_comparege4sp(__m128 left, __m128 right)
{
  return (__m128) vec_cmpge ((vector float) left, (vector float) right);
}

/* Compare 4 32-bit floats for !>= to mask */
VECLIB_INLINE __m128 vec_comparenotge4sp(__m128 left, __m128 right)
{
  __m128_union leftx;
  leftx.as_m128 = vec_comparege4sp (left, right);
  __m128_union rightx;
  rightx.as_m128 = (__m128) vec_splats ((unsigned char) 0xFF);
  return (__m128) vec_xor ((vector float) leftx.as_m128, (vector float) rightx.as_m128);
}

/* Compare scalar 32-bit floats for >= to mask keeping upper elements */
VECLIB_INLINE __m128 vec_comparegelower1of4sp (__m128 left, __m128 right)
{
  vector unsigned int select_vector = {
    0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
  };
  vector float inter_res = vec_comparege4sp ((vector float) left, (vector float) right);
  vector float result = vec_sel ((vector float) left, inter_res, select_vector);
  return (__m128) result;
}

/* Compare scalar 32-bit floats for !>= to mask keeping upper elements */
VECLIB_INLINE __m128 vec_comparenotgelower1of4sp (__m128 left, __m128 right)
{
  vector unsigned int select_vector = {
    0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
  };
  vector float inter_res = vec_comparenotge4sp ((vector float) left, (vector float) right);
  vector float result = vec_sel ((vector float) left, inter_res, select_vector);
  return (__m128) result;
}

/* Compare scalar 32-bit floats for >= to bool */
VECLIB_INLINE int vec_comparege1of4sptobool (__m128 left, __m128 right)
{
  __m128_union res_union;
  __m128_union nan_union;
  res_union.as_m128 = vec_comparege4sp(left, right);
  nan_union.as_m128 = vec_comparenotnans4sp(left, right);
  /* Return 0s for NaN */
  return nan_union.as_hex_unsigned[0] == 0xFFFFFFFFu && res_union.as_hex_unsigned[0] == 0xFFFFFFFFu;
}

/* Compare scalar 32-bit floats for >= to bool QNaNs nonsignaling */
VECLIB_INLINE int vec_comparege1of4sptoboolnonsignaling (__m128 left, __m128 right)
{
  __m128_union res_union;
  __m128_union nan_union;
  res_union.as_m128 = vec_comparege4sp(left, right);
  nan_union.as_m128 = vec_comparenotnans4sp(left, right);
  /* Return 0s for NaN */
  return nan_union.as_hex_unsigned[0] == 0xFFFFFFFFu && res_union.as_hex_unsigned[0] == 0xFFFFFFFFu;
}

/* Compare Ordered / Unordered */

/* Compare 4 32-bit floats for not NaNs to mask */
VECLIB_INLINE __m128 vec_comparenotnans4sp (__m128 left, __m128 right)
{
  __m128 left_mask = vec_compareeq4sp(left, left);
  __m128 right_mask = vec_compareeq4sp(right, right);
  return (__m128) vec_and (left_mask, right_mask);
}

/* Compare 4 32-bit floats for NaNs to mask */
VECLIB_INLINE __m128 vec_comparenans4sp (__m128 left, __m128 right)
{
  __m128_union leftx;
  leftx.as_m128 = vec_comparenotnans4sp (left, right);
  __m128_union rightx;
  rightx.as_m128 = (__m128) vec_splats ((unsigned char) 0xFF);
  return (__m128) vec_xor ((vector float) leftx.as_m128, (vector float) rightx.as_m128);
}

/* Compare scalar 32-bit floats for not NaNs to mask keeping upper elements */
VECLIB_INLINE __m128 vec_comparenotnanslower1of4sp (__m128 left, __m128 right)
{
  vector unsigned int select_vector = {
    0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
  };
  vector float inter_res = vec_comparenotnans4sp ((vector float) left, (vector float) right);
  vector float result = vec_sel ((vector float) left, inter_res, select_vector);
  return (__m128) result;
}

/* Compare scalar 32-bit floats for NaNs to mask keeping upper elements */
VECLIB_INLINE __m128 vec_comparenanslower1of4sp (__m128 left, __m128 right)
{
  vector unsigned int select_vector = {
    0xFFFFFFFFu, 0x00000000u, 0x00000000u, 0x00000000u
  };
  vector float inter_res = vec_comparenans4sp ((vector float) left, (vector float) right);
  vector float result = vec_sel ((vector float) left, inter_res, select_vector);
  return (__m128) result;
}

/******************************************************* Cast *********************************************************/

/* Cast __128i to __m128 */
VECLIB_INLINE __m128 vec_cast1qto4sp (__m128i v)
{
  __m128_all_union v_union;
  v_union.as_m128i = v;
  return (__m128) v_union.as_m128;
}

/* Cast lower 4 of 8 32-bit floats to 4 32-bit floats */
#ifdef VECLIB_VSX
  VECLIB_INLINE __m128 vec_cast4of8spto4sp (__m256 from)
  {
    __m256_union from_union;  from_union.as_m256 = from;
  #ifdef __LITTLE_ENDIAN__
    return from_union.as_m128[0];
  #elif __BIG_ENDIAN__
    return from_union.as_m128[1];
  #endif
  }
#endif

/* Cast __m128d to __m128 */
VECLIB_INLINE __m128 vec_Cast2dpto4sp (__m128d from) {
  __m128_all_union newFrom; newFrom.as_m128d = from;
  return newFrom.as_m128;
}

/********************************************************* Mathematics ************************************************/

/* Add 2 odd and subtract 2 even 32-bit floats - (A0-B0, A1+B1, A2-B2, A3+B3) */
VECLIB_INLINE __m128 vec_addsub4sp (__m128 left, __m128 right) {
  __m128_all_union negation;
  negation.as_vector_unsigned_int = (vector unsigned int) {
    #ifdef __LITTLE_ENDIAN__
      0x80000000u, 0x00000000u, 0x80000000u, 0x00000000u
    #elif __BIG_ENDIAN__
      0x00000000u, 0x80000000u, 0x00000000u, 0x80000000u
    #endif
  };
  __m128 tempResult = (vector float) vec_xor (right, negation.as_vector_float);
  return (vector float) vec_add (left, tempResult);
}

/* Horizontally add 2+2 adjacent pairs of 32-bit floats to 4 32-bit floats - (B0 + B1, B2 + B3, A0 + A1, A2 + A3) */
  VECLIB_INLINE __m128 vec_horizontaladd4sp (__m128 lower, __m128 upper) {
    vector unsigned char transformation2 = {
      #ifdef __LITTLE_ENDIAN__
        0x00, 0x01, 0x02, 0x03,  0x08, 0x09, 0x0A, 0x0B,  0x10, 0x11, 0x12, 0x13,   0x18, 0x19, 0x1A, 0x1B
      #elif __BIG_ENDIAN__
        0x14, 0x15, 0x16, 0x17,  0x1C, 0x1D, 0x1E, 0x1F,  0x04, 0x05, 0x06, 0x07,   0x0C, 0x0D, 0x0E, 0x0F
      #endif
      };
    vector unsigned char transformation1 = {
      #ifdef __LITTLE_ENDIAN__
        0x04, 0x05, 0x06, 0x07,  0x0C, 0x0D, 0x0E, 0x0F,  0x14, 0x15, 0x16, 0x17,    0x1C, 0x1D, 0x1E, 0x1F
      #elif __BIG_ENDIAN__
        0x10, 0x11, 0x12, 0x13,  0x18, 0x19, 0x1A, 0x1B,  0x00, 0x01, 0x02, 0x03,    0x08, 0x09, 0x0A, 0x0B
      #endif
      };
    return (vector float) vec_add (vec_perm ((vector float) lower, (vector float) upper, transformation1),
                                   vec_perm ((vector float) lower, (vector float) upper, transformation2));
  }

/* Horizontally subtract lower pairs then upper pairs of 32-bit floats - (A0-A1, A2-A3, B0-B1, B2-B3) */
VECLIB_INLINE __m128 vec_horizontalsub4sp (__m128 lower, __m128 upper) {
  vector unsigned char transformation2 = {
    #ifdef __LITTLE_ENDIAN__
      0x00, 0x01, 0x02, 0x03,  0x08, 0x09, 0x0A, 0x0B,  0x10, 0x11, 0x12, 0x13,   0x18, 0x19, 0x1A, 0x1B
    #elif __BIG_ENDIAN__
      0x10, 0x11, 0x12, 0x13,  0x18, 0x19, 0x1A, 0x1B,  0x00, 0x01, 0x02, 0x03,    0x08, 0x09, 0x0A, 0x0B
    #endif
  };
  vector unsigned char transformation1 = {
    #ifdef __LITTLE_ENDIAN__
      0x04, 0x05, 0x06, 0x07,  0x0C, 0x0D, 0x0E, 0x0F,  0x14, 0x15, 0x16, 0x17,    0x1C, 0x1D, 0x1E, 0x1F
    #elif __BIG_ENDIAN__
      0x14, 0x15, 0x16, 0x17,  0x1C, 0x1D, 0x1E, 0x1F,  0x04, 0x05, 0x06, 0x07,   0x0C, 0x0D, 0x0E, 0x0F
    #endif
  };
  return (vector float) vec_sub (vec_perm ((vector float) lower, (vector float) upper, transformation2),
                                 vec_perm ((vector float) lower, (vector float) upper, transformation1));
}

/******************************************************** Unpack ******************************************************/

/* Unpack 2+2 32-bit floats from high halves and interleave */
VECLIB_INLINE __m128 vec_unpackupper2spto4sp (__m128 even, __m128 odd)
{
  static const vector unsigned char permute_selector = {
    #ifdef __LITTLE_ENDIAN__
      0x08, 0x09, 0x0A, 0x0B,  0x18, 0x19, 0x1A, 0x1B,  0x0C, 0x0D, 0x0E, 0x0F,  0x1C, 0x1D, 0x1E, 0x1F
    #elif __BIG_ENDIAN__
      0x10, 0x11, 0x12, 0x13,  0x00, 0x01, 0x02, 0x03,  0x14, 0x15, 0x16, 0x17,  0x04, 0x05, 0x06, 0x07
    #endif
  };
  return vec_perm (even, odd, permute_selector);
}

/* Unpack 2+2 32-bit floats from high halves and interleave - deprecated - use previous function */
VECLIB_INLINE __m128 vec_unpackupper22spto4sp (__m128 even, __m128 odd)
{
  return vec_unpackupper2spto4sp (even, odd);
}

/* Unpack 2+2 32-bit floats from low halves and interleave */
VECLIB_INLINE __m128 vec_unpacklower2spto4sp (__m128 even, __m128 odd)
{
  static const vector unsigned char permute_selector = {
    #ifdef __LITTLE_ENDIAN__
      0x00, 0x01, 0x02, 0x03,  0x10, 0x11, 0x12, 0x13,  0x04, 0x05, 0x06, 0x07,  0x14, 0x15, 0x16, 0x17
    #elif __BIG_ENDIAN__
      0x18, 0x19, 0x1A, 0x1B,  0x08, 0x09, 0x0A, 0x0B,  0x1C, 0x1D, 0x1E, 0x1F,  0x0C, 0x0D, 0x0E, 0x0F
    #endif
  };
  return vec_perm (even, odd, permute_selector);
}

/* Unpack 2+2 32-bit floats from low halves and interleave - deprecated - use previous function */
VECLIB_INLINE __m128 vec_unpacklower22spto4sp (__m128 even, __m128 odd)
{
  return vec_unpacklower2spto4sp (even, odd);
}

#endif
