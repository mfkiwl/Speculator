#ifndef __xfft_hop_h__
#define __xfft_hop_h__

#include"xfft_kernel.h"

void hfft_bki( xfft_kernel_t* const __restrict, CUtexref* __restrict, CUmodule, CUdeviceptr, int, int, int );

#endif