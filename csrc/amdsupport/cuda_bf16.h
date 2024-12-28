#pragma once
#ifdef __HIPCC__
#include <hip/hip_bf16.h>
#else
#include_next <cuda_bf16.h>
#endif // __HIPCC__