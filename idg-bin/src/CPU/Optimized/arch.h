#if defined(__INTEL_COMPILER)

#include <immintrin.h>

int has_intel_knl_features()
{
    const unsigned long knl_features =
        (_FEATURE_AVX512F | _FEATURE_AVX512ER |
         _FEATURE_AVX512PF | _FEATURE_AVX512CD );
    return _may_i_use_cpu_feature( knl_features );
}

int check_4th_gen_intel_core_features()
{
    const int the_4th_gen_features =
        (_FEATURE_AVX2 | _FEATURE_FMA | _FEATURE_BMI | _FEATURE_LZCNT | _FEATURE_MOVBE);
    return _may_i_use_cpu_feature( the_4th_gen_features );
}

#else /* non-Intel compiler */

#include <stdint.h>

void run_cpuid(uint32_t eax, uint32_t ecx, uint32_t* abcd)
{
    #if defined(__x86_64__)
    uint32_t ebx, edx;
    __asm__ ( "cpuid" : "+b" (ebx), "+a" (eax), "+c" (ecx), "=d" (edx) );
    abcd[0] = eax; abcd[1] = ebx; abcd[2] = ecx; abcd[3] = edx;
    #endif
}

int check_xcr0_ymm()
{
    uint32_t xcr0;
    #if defined(__x86_64__)
    __asm__ ("xgetbv" : "=a" (xcr0) : "c" (0) : "%edx" );
    #endif

    return ((xcr0 & 6) == 6); /* checking if xmm and ymm state are enabled in XCR0 */
}

int check_xcr0_zmm()
{
    uint32_t xcr0;
    uint32_t zmm_ymm_xmm = (7 << 5) | (1 << 2) | (1 << 1);
    #if defined(__x86_64__)
    __asm__ ("xgetbv" : "=a" (xcr0) : "c" (0) : "%edx" );
    #endif
    return ((xcr0 & zmm_ymm_xmm) == zmm_ymm_xmm); /* check if xmm, zmm and zmm state are enabled in XCR0 */
}

int check_4th_gen_intel_core_features()
{
    uint32_t abcd[4];
    uint32_t fma_movbe_osxsave_mask = ((1 << 12) | (1 << 22) | (1 << 27));
    uint32_t avx2_bmi12_mask = (1 << 5) | (1 << 3) | (1 << 8);

    /* CPUID.(EAX=01H, ECX=0H):ECX.FMA[bit 12]==1   &&
       CPUID.(EAX=01H, ECX=0H):ECX.MOVBE[bit 22]==1 &&
       CPUID.(EAX=01H, ECX=0H):ECX.OSXSAVE[bit 27]==1 */
    run_cpuid( 1, 0, abcd );
    if ( (abcd[2] & fma_movbe_osxsave_mask) != fma_movbe_osxsave_mask )
        return 0;

    #if 0
    if ( ! check_xcr0_ymm() )
        return 0;
    #endif

    /*  CPUID.(EAX=07H, ECX=0H):EBX.AVX2[bit 5]==1  &&
        CPUID.(EAX=07H, ECX=0H):EBX.BMI1[bit 3]==1  &&
        CPUID.(EAX=07H, ECX=0H):EBX.BMI2[bit 8]==1  */
    run_cpuid( 7, 0, abcd );
    if ( (abcd[1] & avx2_bmi12_mask) != avx2_bmi12_mask )
        return 0;

    /* CPUID.(EAX=80000001H):ECX.LZCNT[bit 5]==1 */
    run_cpuid( 0x80000001, 0, abcd );
    if ( (abcd[2] & (1 << 5)) == 0)
        return 0;

    return 1;
}

int has_intel_knl_features() {
  uint32_t abcd[4];
  uint32_t osxsave_mask = (1 << 27); // OSX.
  uint32_t avx2_bmi12_mask = (1 << 16) | // AVX-512F
                             (1 << 26) | // AVX-512PF
                             (1 << 27) | // AVX-512ER
                             (1 << 28);  // AVX-512CD
  run_cpuid( 1, 0, abcd );
  // step 1 - must ensure OS supports extended processor state management
  if ( (abcd[2] & osxsave_mask) != osxsave_mask )
    return 0;
  // step 2 - must ensure OS supports ZMM registers (and YMM, and XMM)
  if ( ! check_xcr0_zmm() )
    return 0;

  return 1;
}

#endif /* non-Intel compiler */
