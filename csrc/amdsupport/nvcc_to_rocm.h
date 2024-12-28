#pragma once
#include <stdint.h>
#include <hip/amd_detail/amd_hip_complex.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/amd_detail/amd_warp_sync_functions.h>
#include <hip/amd_detail/hip_ldg.h>

typedef hipFloatComplex cuFloatComplex;
typedef hipDoubleComplex cuDoubleComplex;

constexpr auto cuCreal = hipCreal;
constexpr auto cuCimag = hipCimag;

constexpr auto cuCrealf = hipCrealf;
constexpr auto cuCimagf = hipCimagf;

constexpr auto make_cuFloatComplex = make_hipFloatComplex;
constexpr auto make_cuDoubleComplex = make_hipDoubleComplex;

typedef __hip_bfloat16 __nv_bfloat16;
typedef __hip_bfloat16_raw __nv_bfloat16_raw;
typedef __hip_fp8x2_e4m3_fnuz __nv_fp8x2_e4m3;

typedef __hip_fp8x2_e5m2_fnuz __nv_fp8x2_e5m2;

typedef __hip_saturation_t __nv_saturation_t;
#define __NV_NOSAT __HIP_NOSAT
#define __NV_SATFINITE __HIP_SATFINITE

typedef __hip_fp8_interpretation_t __nv_fp8_interpretation_t;
#define __NV_E4M3 __HIP_E4M3_FNUZ
#define __NV_E5M2 __HIP_E5M2_FNUZ

constexpr auto __nv_cvt_float2_to_fp8x2 =__hip_cvt_float2_to_fp8x2;

constexpr int min_constexpr(int a, int b) {
    return (a < b) ? a : b;
}


#define CUTE_HOST_DEVICE __forceinline__ __host__ __device__
#define CUTE_DEVICE      __forceinline__          __device__
#define CUTE_HOST        __forceinline__ __host__


#define __CUDA__


__device__ inline static __half __ldg(const __half* ptr) { return *ptr; }

