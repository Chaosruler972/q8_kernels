import subprocess
import os
import multiprocessing
from packaging.version import parse, Version
from pathlib import Path
from setuptools import setup, find_packages
from torch.utils.cpp_extension import (
    BuildExtension,
    CUDAExtension,
    CUDA_HOME,
)

# package name managed by pip, which can be remove by `pip uninstall tiny_pkg`
PACKAGE_NAME = "q8_kernels"

ext_modules = []
generator_flag = []
cc_flag = []



# Find better way to find out if it is nvidia or amd
is_nvidia = False
is_amd = True


if is_nvidia:
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_89,code=sm_89")

cuda_flags = [
    '--ptxas-options=-v',
    '--ptxas-options=-O2',
    '--expt-relaxed-constexpr',
    '--expt-extended-lambda',
    '--use_fast_math',
    "-lineinfo",
]

rocm_flags = [
    '-I/opt/rocm/include',
    '-I/opt/rocm/llvm/include/',
    '-DHIP_ENABLE_WARP_SYNC_BUILTINS=1',
]

chosen_flags = cuda_flags if is_nvidia else rocm_flags



# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

nvidia_includes = [
    Path('/usr') / 'local' / 'cuda' / 'include',
    Path(this_dir) / "third_party/cutlass/include",
    Path(this_dir) / "third_party/cutlass/tools/utils/include" ,
    Path(this_dir) / "third_party/cutlass/examples/common" ,
]

rocm_includes = [
    Path('/opt') / 'rocm' / 'include',
    Path(this_dir) / "third_party/cutlass_hip/include",
    Path(this_dir) / "third_party/cutlass_hip/tools/utils/include" ,
    Path(this_dir) / "third_party/cutlass_hip/examples/common" ,
]

chosen_includes = nvidia_includes if is_nvidia else rocm_includes


# helper function to get cuda version
def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


if CUDA_HOME is not None:
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version >= Version("11.8"):
        if is_nvidia:
            cc_flag.append("-gencode")
            cc_flag.append("arch=compute_89,code=sm_89")        


# cuda module
ext_modules.append(
    CUDAExtension(
        # package name for import
        name="q8_kernels_cuda.gemm._C",
        sources=[
            "csrc/gemm/q8_gemm_api.cpp",
            "csrc/gemm/q8_matmul_bias.cu",
            "csrc/gemm/q8_matmul.cu",
        ],
        extra_compile_args={
            # add c compile flags
            "cxx": ["-O3", "-std=c++17"] + generator_flag,
            # add nvcc compile flags
            "nvcc": [
                    *chosen_flags,
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                ]
                + generator_flag
                + cc_flag,
        },
        include_dirs=[
            *chosen_includes,
            Path(this_dir) / "csrc" / "amdsupport",
            Path(this_dir) / "csrc" / "gemm",
        ],
    )
)

ext_modules.append(
    CUDAExtension(
        # package name for import
        name="q8_kernels_cuda.quantizer._C",
        sources=[
            "csrc/quantizer/tokenwise_quant.cpp",
            "csrc/quantizer/tokenwise_quant.cu",
        ],
          extra_compile_args={
            # add c compile flags
            "cxx": ["-O3", "-std=c++17"] + generator_flag,
            # add nvcc compile flags
            "nvcc": [
                    *chosen_flags,
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                ]
                + generator_flag
                + cc_flag,
        },
        include_dirs=[
            *chosen_includes,
            Path(this_dir) / "csrc" / "amdsupport",
            Path(this_dir) / "csrc"/"quantizer",
        ],
    )
)


ext_modules.append(
    CUDAExtension(
        # package name for import
        name="q8_kernels_cuda.ops._C",
        sources=[
            "csrc/ops/ops_api.cpp",
            
            "csrc/ops/rope.cpp",
            "csrc/ops/rope.cu",
            
            "csrc/ops/rms_norm.cpp",
            "csrc/ops/rms_norm.cu",

            "csrc/ops/fma.cpp",
            "csrc/ops/fma.cu",

            "csrc/fast_hadamard/fast_hadamard_transform.cpp",
            "csrc/fast_hadamard/fast_hadamard_transform.cu"
        ],
          extra_compile_args={
            # add c compile flags
            "cxx": ["-O3", "-std=c++17"] + generator_flag,
            # add nvcc compile flags
            "nvcc": [
                    *chosen_flags,
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                ]
                + generator_flag
                + cc_flag,
        },
        include_dirs=[
            *chosen_includes,
            Path(this_dir) / "csrc" / "amdsupport",
            Path(this_dir) / "csrc"/"ops",
            Path(this_dir) / "csrc"/"fast_hadamard",
        ],
    )
)


ext_modules.append(
    CUDAExtension(
        # package name for import
        name="q8_kernels_cuda.flash_attention._C",
        sources=[
            "csrc/flash_attention/flash_attention.cpp",
            "csrc/flash_attention/flash_attention.cu",
            "csrc/flash_attention/flash_attention_mask.cu",
        ],
          extra_compile_args={
            # add c compile flags
            "cxx": ["-O3", "-std=c++17"] + generator_flag,
            # add nvcc compile flags
            "nvcc": [
                    *chosen_flags,
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                ]
                + generator_flag
                + cc_flag,
        },
        include_dirs=[
            *chosen_includes,
            Path(this_dir) / "csrc" / "amdsupport",
            Path(this_dir) / "csrc"/"flash_attention",
        ],
    )
)

setup(
    name=PACKAGE_NAME,
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
        )
    ),
    description="8bit kernels",
    ext_modules=ext_modules,
    entry_points={
        'console_scripts': [
            'q8_kernels.convert_weights=q8_kernels.utils.convert_weights:main',
        ],
    },
    cmdclass={ "build_ext": BuildExtension},
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "einops",
        "packaging",
        "ninja",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    author="KONAKONA666/Aibek Bekbayev",
    author_email="konakona666@proton.me",
)




