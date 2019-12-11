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

#ifndef _H_VECMISC
#define _H_VECMISC

#include <stdio.h>
#include <altivec.h>
#include "veclib_types.h"

/************************************************* Floating-Point Control and Status Register *************************/

/* Get exception mask bits from MXCSR register */
VECLIB_INLINE unsigned int vec_getfpexceptionmask (void)
{
  #ifdef __ibmxl__
    unsigned long long exception_mask_mask = 0x1F000002;
    double FPSCR_content = __readflm();
    unsigned long long *upper_32bits = (unsigned long long *) &FPSCR_content;
    *upper_32bits = *upper_32bits & 0xFFFFFFFF;
    return (unsigned int) (*upper_32bits & exception_mask_mask);
  #else
    /* Do nothing for now */
    return (0ull);
  #endif
}

/* Get exception state bits from MXCSR register */
VECLIB_INLINE unsigned int vec_getfpexceptionstate (void)
{
  #ifdef __ibmxl__
    unsigned long long exception_state_mask = 0xE01FFD;
    double FPSCR_content = __readflm();
    unsigned long long *upper_32bits = (unsigned long long *) &FPSCR_content;
    *upper_32bits = *upper_32bits & 0xFFFFFFFF;
    return (unsigned int) (*upper_32bits & exception_state_mask);
  #else
    /* Do nothing for now */
    return (0u);
  #endif
}

/* Get flush zero bits from MXCSR register */
VECLIB_INLINE unsigned int vec_getfpflushtozeromode (void)
{
  /* Do nothing for now */
  return 0u;
}

/* Get rounding mode bits from MXCSR register */
VECLIB_INLINE unsigned int vec_getfproundingmode (void)
{
  #ifdef __ibmxl__
    unsigned long long rounding_mode_mask = 0x00000003;
    double FPSCR_content = __readflm();
    unsigned long long *upper_32bits = (unsigned long long *) &FPSCR_content;
    *upper_32bits = *upper_32bits & 0xFFFFFFFF;
    return (unsigned int) (*upper_32bits & rounding_mode_mask);
  #else
    /* Do nothing for now */
    return 0u;
  #endif
}

/* Get all bits from MXCSR register */
VECLIB_INLINE unsigned int vec_getfpallbits (void)
{
  #ifdef __ibmxl__
    unsigned long long all_bits_mask = 0xFFFFFFFF;
    double FPSCR_content = __readflm();
    unsigned long long *upper_32bits = (unsigned long long *) &FPSCR_content;
    *upper_32bits = *upper_32bits & 0xFFFFFFFF;
    return (unsigned int) (*upper_32bits & all_bits_mask);
  #else
    /* Do nothing for now */
    return 0u;
  #endif
}

/* Set exception mask bits in MXCSR register */
VECLIB_INLINE void vec_setfpexceptionmask (unsigned int mask)
{
  #ifdef __ibmxl__
    unsigned long long exception_mask_mask = 0x1F000002;
    unsigned long long exception_mask = (unsigned long long) mask & exception_mask_mask;
    double *exception_mask_bits;
    exception_mask_bits  = (double*) &exception_mask;
    __setflm (*exception_mask_bits);
  #else
    /* Do nothing for now */
  #endif
}

/* Set exception state bits in MXCSR register */
VECLIB_INLINE void vec_setfpexceptionstate (unsigned int mask)
{
  #ifdef __ibmxl__
    unsigned long long exception_state_mask = 0xE01FFD;
    unsigned long long exception_state = (unsigned long long) mask & exception_state_mask;
    double* exception_state_bits;
    exception_state_bits = (double*) &exception_state;
    __setflm (*exception_state_bits);
  #else
    /* Do nothing for now */
  #endif
}

/* Set flush zero bits in MXCSR register */
VECLIB_INLINE void vec_setfpflushtozeromode (unsigned int mask)
{
  /* Do nothing for now */
}

/* Set rounding mode bits in MXCSR register */
VECLIB_INLINE void vec_setfproundingmode (unsigned int mask)
{
  #ifdef __ibmxl__
    unsigned int rounding_mode_mask = 0x3;
    unsigned int rounding_mode = mask & rounding_mode_mask;
    __setrnd (rounding_mode);
  #else
    /* Do nothing for now */
  #endif
}

/* Set all bits in MXCSR register */
VECLIB_INLINE void vec_setfpallbits (unsigned int mask)
{
  #ifdef __ibmxl__
    /* Set bits from 63:32 */
    unsigned long long all_bits_mask = 0xFFFFFFFF;
    unsigned long long all_bits = (unsigned long long) mask & all_bits_mask;
    double* fpscr_all_bits = (double*) &all_bits;
    __setflm (*fpscr_all_bits);
  #else
    /* Do nothing for now */
  #endif
}

/*************************************************** Malloc/Free ******************************************************/

/* Allocate aligned vector memory block */
VECLIB_INLINE void* vec_malloc (size_t size, size_t align) {

    void *result;
    #ifdef _MSC_VER
    result = _aligned_malloc(size, align);
    #else
     if (posix_memalign (&result, align, size)) result = 0;
    #endif
    return result;
}

/* Free aligned vector memory block */
VECLIB_INLINE void vec_free (void *ptr) {

    #ifdef _MSC_VER
        _aligned_free(ptr);
    #else
      free(ptr);
    #endif
}

/* Pause spin-wait loop */
VECLIB_INLINE void vec_pause (void) { }

/* Serialize previous loads and stores before following loads and stores */
VECLIB_INLINE void vec_fencestoreloads (void) {
  #ifdef __ibmxl__
    __sync();
  #else
    __atomic_thread_fence( 5 );
  #endif
}

/***************************************************** Boolean ********************************************************/

/* Population Count unsigned long long */
VECLIB_INLINE int vec_popcount1uw (unsigned long long a) {
  int result;
  #ifdef __ibmxl__
    result = __popcnt8 (a);
  #else
    /* gcc */
    asm("   popcntd %0, %1"
    :   "=r"     (result)
    :   "r"      (a)
    );
  #endif
  return result;
}

/******************************************************* CRC **********************************************************/

  #ifdef __LITTLE_ENDIAN__
    static const int upper64 = 1;  /* subscript for upper half of vector register */
    static const int lower64 = 0;  /* subscript for lower half of vector register */
  #elif __BIG_ENDIAN__
    static const int upper64 = 0;  /* subscript for upper half of vector register */
    static const int lower64 = 1;  /* subscript for lower half of vector register */
  #endif

  #ifdef __LITTLE_ENDIAN__
    static const int upper32       = 3;  /* subscript for upper quarter of vector register */
    static const int uppermiddle32 = 2;  /* subscript for upper middle quarter of vector register */
    static const int lowermiddle32 = 1;  /* subscript for lower middle quarter of vector register */
    static const int lower32       = 0;  /* subscript for lower quarter of vector register */
  #elif __BIG_ENDIAN__
    static const int upper32       = 0;  /* subscript for upper quarter of vector register */
    static const int uppermiddle32 = 1;  /* subscript for upper middle quarter of vector register */
    static const int lowermiddle32 = 2;  /* subscript for lower middle quarter of vector register */
    static const int lower32       = 3;  /* subscript for lower quarter of vector register */
  #endif

static inline unsigned long long veclib_bitreverse16 (unsigned short input)
{
  /* Reverse the bits of a 16-bit input */
  unsigned long long source = (unsigned long long) input;

  /* Bit permute selector for reversing bytes of the input */
  /* Bit numbering is 0...63 left to right.  Byte numbering is 0...7 left to right. */
  const unsigned long long reverse_of_byte_6_selector = 0x3736353433323130;
  const unsigned long long reverse_of_byte_7_selector = 0x3F3E3D3C3B3A3938;

  /* Reverse bytes of the input, each result is in the lower 8 bits and zero filled */
  #ifdef __ibmxl__
    /* xlc */
    unsigned long long reverse_of_byte_6 = __bpermd (reverse_of_byte_6_selector, source);
    unsigned long long reverse_of_byte_7 = __bpermd (reverse_of_byte_7_selector, source);
  #else
    /* gcc */
    unsigned long long reverse_of_byte_6 = __builtin_bpermd (reverse_of_byte_6_selector, source);
    unsigned long long reverse_of_byte_7 = __builtin_bpermd (reverse_of_byte_7_selector, source);
  #endif

  /* Concatenate reversed bytes in reverse order */
  unsigned long long result = (reverse_of_byte_7 << 8) | reverse_of_byte_6;
  return result;
}

static inline unsigned long long veclib_bitreverse32 (unsigned int input)
{
  /* Reverse the bits of a 32-bit input */
  long long source = (long long) input;
  unsigned long long result = 0;

  /* Bit permute selector for reversing 0th..3rd byte of the input */
  const long long bit_selector0 = 0x3F3E3D3C3B3A3938;
  const long long bit_selector1 = 0x3736353433323130;
  const long long bit_selector2 = 0x2F2E2D2C2B2A2928;
  const long long bit_selector3 = 0x2726252423222120;

  /* Reverse 0th..3rd byte of the input, the result is in the lower 8 bits */
  #ifdef __ibmxl__
    /* xlc */
    long long reverse_byte0 = __bpermd (bit_selector0, source);
    long long reverse_byte1 = __bpermd (bit_selector1, source);
    long long reverse_byte2 = __bpermd (bit_selector2, source);
    long long reverse_byte3 = __bpermd (bit_selector3, source);
  #else
    /* gcc */
    long long reverse_byte0 = __builtin_bpermd (bit_selector0, source);
    long long reverse_byte1 = __builtin_bpermd (bit_selector1, source);
    long long reverse_byte2 = __builtin_bpermd (bit_selector2, source);
    long long reverse_byte3 = __builtin_bpermd (bit_selector3, source);
  #endif

  /* Rotate and insert reverse_byte0..3 to the 3rd..0th byte of the result */
  #ifdef __ibmxl__
    /* xlc */
    result = __rlwimi ((unsigned int) reverse_byte0, (unsigned int) result, 24, 0xFF000000);
    result = __rlwimi ((unsigned int) reverse_byte1, (unsigned int) result, 16, 0xFF0000);
    result = __rlwimi ((unsigned int) reverse_byte2, (unsigned int) result, 8, 0xFF00);
    result = __rlwimi ((unsigned int) reverse_byte3, (unsigned int) result, 0, 0xFF);  
  #else
    reverse_byte0 = (reverse_byte0 << 24) & 0xFF000000;
    reverse_byte1 = (reverse_byte1 << 16) & 0xFF0000;
    reverse_byte2 = (reverse_byte2 << 8) & 0xFF00;
    reverse_byte3 = (reverse_byte3 << 0) & 0xFF;
    unsigned long long reverse_byte01 = reverse_byte0 | reverse_byte1;
    unsigned long long reverse_byte23 = reverse_byte2 | reverse_byte3;
    result = reverse_byte01 | reverse_byte23;
  #endif

  return result;
}

static inline unsigned long long veclib_bitreverse64 (unsigned long long source)
{
  /* Reverse the bits of a 64-bit input */

  /* Bit permute selectors for reversing bytes of the input */
  /* Bit numbering is 0...63 left to right.  Byte numbering is 0...7 left to right. */
  const long long reverse_of_byte_0_selector = 0x0706050403020100;  /* bits  7...0  */
  const long long reverse_of_byte_1_selector = 0x0F0E0D0C0B0A0908;  /* bits 15...8  */
  const long long reverse_of_byte_2_selector = 0x1716151413121110;  /* bits 23...16 */
  const long long reverse_of_byte_3_selector = 0x1F1E1D1C1B1A1918;  /* bits 31...24 */
  const long long reverse_of_byte_4_selector = 0x2726252423222120;  /* bits 39...32 */
  const long long reverse_of_byte_5_selector = 0x2F2E2D2C2B2A2928;  /* bits 47...40 */
  const long long reverse_of_byte_6_selector = 0x3736353433323130;  /* bits 55...48 */
  const long long reverse_of_byte_7_selector = 0x3F3E3D3C3B3A3938;  /* bits 63...56 */

  /* Reverse bytes of the input, each result is in the lower 8 bits zero filled */
  #ifdef __ibmxl__
    /* xlc */
    long long reverse_of_byte_0 = __bpermd (reverse_of_byte_0_selector, source);
    long long reverse_of_byte_1 = __bpermd (reverse_of_byte_1_selector, source);
    long long reverse_of_byte_2 = __bpermd (reverse_of_byte_2_selector, source);
    long long reverse_of_byte_3 = __bpermd (reverse_of_byte_3_selector, source);
    long long reverse_of_byte_4 = __bpermd (reverse_of_byte_4_selector, source);
    long long reverse_of_byte_5 = __bpermd (reverse_of_byte_5_selector, source);
    long long reverse_of_byte_6 = __bpermd (reverse_of_byte_6_selector, source);
    long long reverse_of_byte_7 = __bpermd (reverse_of_byte_7_selector, source);
  #else
    /* gcc */
    long long reverse_of_byte_0 = __builtin_bpermd (reverse_of_byte_0_selector, source);
    long long reverse_of_byte_1 = __builtin_bpermd (reverse_of_byte_1_selector, source);
    long long reverse_of_byte_2 = __builtin_bpermd (reverse_of_byte_2_selector, source);
    long long reverse_of_byte_3 = __builtin_bpermd (reverse_of_byte_3_selector, source);
    long long reverse_of_byte_4 = __builtin_bpermd (reverse_of_byte_4_selector, source);
    long long reverse_of_byte_5 = __builtin_bpermd (reverse_of_byte_5_selector, source);
    long long reverse_of_byte_6 = __builtin_bpermd (reverse_of_byte_6_selector, source);
    long long reverse_of_byte_7 = __builtin_bpermd (reverse_of_byte_7_selector, source);
  #endif

  /* Concatenate reversed bytes in reverse order */
  unsigned long long result = (((reverse_of_byte_7 << 56) | (reverse_of_byte_6 << 48))
                             | ((reverse_of_byte_5 << 40) | (reverse_of_byte_4 << 32)))
                            | (((reverse_of_byte_3 << 24) | (reverse_of_byte_2 << 16))
                             | ((reverse_of_byte_1 << 8)  | (reverse_of_byte_0      )));
  return result;
}

#ifdef _ARCH_PWR8
static inline vector unsigned long long veclib_vec_cntlz_int128 (vector unsigned long long v)
{
  /* Count leading zeros of 128 bit vector */

  vector unsigned long long leading_zeros_per_half =  VEC_CNTLZ ((vector unsigned long long) v);  /* 2 long long counts */
  /* (leading_zeros_per_half[upper] < 64) ? leading_zeros_per_half[upper] : (64 + leading_zeros_per_half[lower]) */
  /* ie (v[upper] != 0) ? leading_zeros_per_half[upper] : (leading_zeros_per_half[upper] + leading_zeros_per_half[lower]) */
  vector unsigned char zeros = vec_xor ((vector unsigned char) v, (vector unsigned char) v);
  vector bool long long v_upper_half_is_zero = vec_cmpeq ((vector unsigned long long) v, (vector unsigned long long) zeros);
  vector unsigned char shifted_mask = vec_sld (zeros, (vector unsigned char) v_upper_half_is_zero, 8);
  vector unsigned long long lower_half_leading_zeros_or_zero = (vector unsigned long long)
      vec_and ((vector unsigned char) leading_zeros_per_half, shifted_mask);
  vector unsigned long long right_justified_upper = (vector unsigned long long)
      vec_sld (zeros, (vector unsigned char) leading_zeros_per_half, 8);
  vector unsigned long long leading_zeros = vec_add (right_justified_upper, lower_half_leading_zeros_or_zero);
  return leading_zeros;
}
#else
/* veclib_vec_cntlz_int128 unsupported until Power8 */
#endif

#ifdef _ARCH_PWR8
static inline unsigned long long veclib_crc_mod2 (unsigned long long dividend)
{
  /* Compute crc32c (Castagnoli) by 64 bit modulo 2 polynomial long division remainder */
  unsigned long long divisor = 0x11EDC6F41;
  if (dividend < 0x100000000) {
    /* Dividend is less than 33 bits */
    return dividend;
  }
  unsigned int leading_zeros = 0;
  #ifdef __ibmxl__
    /* xlc */
    leading_zeros = __cntlz8 (dividend);
  #else
    /* gcc */
    asm("   cntlzd %0, %1"
    :   "=r"     (leading_zeros)
    :   "r"      (dividend)
    );
  #endif
  /* align leftmost bit of divisor with leftmost remaining bit of dividend */
  /* 31 is the number of leading zeros before the divisor = gpr length - divisor length = 64 - 33 */
  divisor = divisor << (31 - leading_zeros);
  dividend = dividend ^ divisor;
  dividend = veclib_crc_mod2 (dividend);
  return dividend;
}
#else
/* veclib_crc_mod2 unsupported until Power8 */
#endif

#ifdef _ARCH_PWR8
static inline vector unsigned char veclib_int96_crc_mod2 (vector unsigned char dividend)
{
  /* Compute crc32c (Castagnoli) by 96 bit modulo 2 polynomial long division remainder */
  /* (Type vector unsigned __int128 or unsigned __int128 would make this much simpler) */

  /* return if done */
  /* if (dividend < 0x100000000) */
  vector unsigned int under_0x100000000 =
    #ifdef __LITTLE_ENDIAN__
      { 0xFFFFFFFF, 0, 0, 0 };
    #elif __BIG_ENDIAN__
      { 0, 0, 0, 0xFFFFFFFF };
    #endif
  if (vec_all_le ((vector unsigned int) dividend, under_0x100000000)) {
    /* Dividend is less than 33 bits so done */
    return dividend;
  }

  unsigned long long crc32c_divisor = 0x11EDC6F41ULL;
  vector unsigned char divisor = (vector unsigned char) (vector unsigned long long)
    #ifdef __LITTLE_ENDIAN__
      { crc32c_divisor, 0ULL };
    #elif __BIG_ENDIAN__
      { 0ULL, crc32c_divisor };
    #endif

  /* unsigned int leading_zeros = vec_cntlz (dividend); */
  vector unsigned long long leading_zeros = veclib_vec_cntlz_int128 ((vector unsigned long long) dividend);

  /* align leftmost bit of divisor with leftmost remaining bit of dividend */
  /* 95 = maximum divisor leading zeros = vr length - divisor length = 128 - 33 = 95 */
  /* divisor = divisor << (95 - leading_zeros); */
  vector unsigned char splatted_maximum_divisor_leading_zeros = (vector unsigned char)
      { 95, 95, 95, 95,  95, 95, 95, 95,  95, 95, 95, 95,  95, 95, 95, 95 };
  #ifdef __LITTLE_ENDIAN__
    #define VECLIB_RIGHTMOST_BYTE 0
  #elif __BIG_ENDIAN__
    #define VECLIB_RIGHTMOST_BYTE 15
  #endif
  vector unsigned char splatted_leading_zeros = vec_splat ((vector unsigned char) leading_zeros, VECLIB_RIGHTMOST_BYTE);
  vector unsigned char shift_count = vec_sub (splatted_maximum_divisor_leading_zeros, splatted_leading_zeros);
  divisor = vec_slo (divisor, shift_count);
  divisor = vec_sll (divisor, shift_count);

  /* dividend = dividend ^ divisor; */
  dividend = vec_xor (dividend, (vector unsigned char) divisor);
  dividend = veclib_int96_crc_mod2 (dividend);
  return dividend;
}
#else
/* veclib_int96_crc_mod2 unsupported until Power8 */
#endif

/* Accumulate CRC32C (Castagnoli) from unsigned char */
VECLIB_INLINE unsigned int vec_crc321ub (unsigned int crc, unsigned char next)
{
  unsigned long long reversed_crc = veclib_bitreverse32 (crc & 0xFFFFFFFF);
  unsigned long long shifted_crc = reversed_crc << 8;

  #ifdef __ibmxl__
    /* xlc */
    unsigned long long reversed_next = (unsigned long long) __bpermd (0x3F3E3D3C3B3A3938, (long long) (next & 0xFF));
  #else
    /* gcc */
    unsigned long long reversed_next = (unsigned long long) __builtin_bpermd (0x3F3E3D3C3B3A3938, (long long) (next & 0xFF));
  #endif
  unsigned long long shifted_next = reversed_next << 32;

  unsigned long long merged = shifted_crc ^ shifted_next;
  unsigned int reversed_new_crc = (unsigned int) (veclib_crc_mod2 (merged) & 0xFFFFFFFF);
  return veclib_bitreverse32 (reversed_new_crc);
}

/* Accumulate CRC32C (Castagnoli) from unsigned short */
VECLIB_INLINE unsigned int vec_crc321uh (unsigned int crc, unsigned short next)
{
  unsigned long long reversed_crc = veclib_bitreverse32 (crc & 0xFFFFFFFF);
  unsigned long long shifted_crc = reversed_crc << 16;

  unsigned long long reversed_next = (unsigned long long) veclib_bitreverse16 ((unsigned short) (next & 0xFFFF));
  unsigned long long shifted_next = reversed_next << 32;

  unsigned long long merged = shifted_crc ^ shifted_next;
  unsigned int reversed_new_crc = (unsigned int) (veclib_crc_mod2 (merged) & 0xFFFFFFFF);
  return veclib_bitreverse32 (reversed_new_crc);
}

/* Accumulate CRC32C (Castagnoli) from unsigned int */
VECLIB_INLINE unsigned int vec_crc321uw (unsigned int crc, unsigned int next)
{
  unsigned long long reversed_crc = veclib_bitreverse32 (crc & 0xFFFFFFFF);
  unsigned long long shifted_crc = reversed_crc << 32;

  unsigned long long reversed_next = veclib_bitreverse32 (next & 0xFFFFFFFF);
  unsigned long long shifted_next = reversed_next << 32;

  unsigned long long merged = shifted_crc ^ shifted_next;
  unsigned int reversed_new_crc = (unsigned int) (veclib_crc_mod2 (merged) & 0xFFFFFFFF);
  return veclib_bitreverse32 (reversed_new_crc);
}

/* Accumulate CRC32 from unsigned int - deprecated - use previous function */
VECLIB_INLINE unsigned int vec_crc324ub (unsigned int crc, unsigned int next) {
  return vec_crc321uw (crc, next);
}

#ifdef _ARCH_PWR8
/* Accumulate CRC32C (Castagnoli) from unsigned long long */
VECLIB_INLINE unsigned long long vec_crc321ud (unsigned long long crc, unsigned long long next)
{
  /* (crc and result should have been unsigned int) */
  unsigned long long reversed_crc = veclib_bitreverse32 (crc & 0xFFFFFFFF);
  /* unsigned int96 shifted_crc = reversed_crc << 64; */
  __m128i_union shifted_crc;
  shifted_crc.as_long_long[upper64] = reversed_crc;
  shifted_crc.as_long_long[lower64] = 0;

  unsigned long long reversed_next = veclib_bitreverse64 (next);
  /* unsigned int96 shifted_next = (unsigned int96) reversed_next << 32; */
  __m128i_union shifted_next;
  shifted_next.as_long_long[upper64] = 0;
  shifted_next.as_long_long[lower64] = reversed_next;
  __m128i_union replicated_count;
  replicated_count.as_m128i = vec_splat16sb (32);
  shifted_next.as_vector_unsigned_char = vec_slo (shifted_next.as_vector_unsigned_char, replicated_count.as_m128i);

  /* unsigned int96 merged = shifted_crc ^ shifted_next; */
  vector unsigned char merged = vec_xor (shifted_crc.as_vector_unsigned_char, shifted_next.as_vector_unsigned_char);

  /* unsigned int reversed_new_crc = (unsigned int) veclib_96_bit_crc_mod2 (merged); */
  __m128i_union int96_crc_mod2;
  int96_crc_mod2.as_vector_unsigned_char = veclib_int96_crc_mod2 ((vector unsigned char) merged);
  unsigned int reversed_new_crc = int96_crc_mod2.as_unsigned_int[lower32];
  return veclib_bitreverse32 (reversed_new_crc);
}
#else
/* vec_crc321ud unsupported until Power8 */
#endif

/****************************************************** Miscellaneous *************************************************/

#define _MM_HINT_T0 0
#define _MM_HINT_T1 0
#define _MM_HINT_T2 0
#define _MM_HINT_NTA 0

/* vec_prefetch hint */
typedef enum vec_prefetch_hint
{
  vec_HINT_NTA = 0,
  vec_HINT_T2  = 1,
  vec_HINT_T1  = 2,
  vec_HINT_T0  = 3
} vec_prefetch_hint;

/* Prefetch cache line with hint */
VECLIB_INLINE void vec_prefetch (void const* address, vec_prefetch_hint hint)
{
  #ifdef __ibmxl__
    __dcbt ((void*) address);
  #else
    /* Do nothing for now */
  #endif
}

/* Zero upper half of all 8 or 16 YMMM registers */
VECLIB_INLINE void vec_zeroallupper (void)
{
  /* Do nothing */
}

/* Serialize previous stores before following stores */
VECLIB_INLINE void vec_fence (void)
{
  #ifdef __ibmxl__
    __fence ();
  #else
    /* Do nothing for now */
  #endif
}

#endif
