#ifdef COMPILER_GCC_X86

/** 
* @file intrinsics.h 
* @brief intrinsics definitions
*
* @author Hendrik Schilling
* @editor Maximilian Diebold
* @date 01/15/2018
*/


#define pand      __builtin_ia32_pand128
#define psubb     __builtin_ia32_psubb128
#define pxor      __builtin_ia32_pxor128
#define pcmpgtb   __builtin_ia32_pcmpgtb128
#define por       __builtin_ia32_por128
#define punpckldq __builtin_ia32_punpckldq128
#define punpckhdq __builtin_ia32_punpckhdq128
#define punpcklwd __builtin_ia32_punpcklwd128
#define punpckhwd __builtin_ia32_punpckhwd128
#define punpcklbw __builtin_ia32_punpcklbw128
#define punpckhbw __builtin_ia32_punpckhbw128
#define pmullw    __builtin_ia32_pmullw128
#define pshufd    __builtin_ia32_pshufd
#define addps     __builtin_ia32_addps
#define subps     __builtin_ia32_subps
#define mulps     __builtin_ia32_mulps
#define paddd     __builtin_ia32_paddd128

#elif COMPILER_CLANG_X86

#include <mmintrin.h>

#define pand      _mm_and_si128
#define psubb     _mm_sub_epi8
#define pxor      _mm_xor_si128
#define pcmpgtb   _mm_cmpgt_epi8
#define por       _mm_or_si128
#define punpckldq _mm_unpacklo_epi32
#define punpckhdq _mm_unpackhi_epi32
#define punpcklwd _mm_unpacklo_epi16
#define punpckhwd _mm_unpackhi_epi16
#define punpcklbw _mm_unpacklo_epi8
#define punpckhbw _mm_unpackhi_epi8
#define pmullw    _mm_mullo_epi16
#define pshufd    _mm_shuffle_epi32
#define addps     _mm_add_ps
#define subps     _mm_sub_ps
#define mulps     _mm_mul_ps
#define paddd     _mm_add_epi64

#endif
