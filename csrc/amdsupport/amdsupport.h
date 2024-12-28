#pragma once
#include <stdint.h>

#ifndef HIP_ENABLE_WARP_SYNC_BUILTINS
#define HIP_ENABLE_WARP_SYNC_BUILTINS
#endif // HIP_ENABLE_WARP_SYNC_BUILTINS

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#include "nvcc_to_rocm.h"
typedef uint64_t shfl_mask_t;
#else
typedef uint32_t shfl_mask_t;
#endif
