#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))

include_dirs=[
            os.path.join(project_root, '../cutlass/include'),
            os.path.join(project_root, '../cutlass/tools/util/include'),
            ]
extra_compile_args={
        'cxx': ['-O3'],
        'nvcc': [
            '-O3',
            '--expt-relaxed-constexpr',
            '-U__CUDA_NO_HALF_OPERATORS__',      # Undefine the disable flag
            '-U__CUDA_NO_HALF_CONVERSIONS__',    # Undefine the disable flag  
            '-U__CUDA_NO_BFLOAT16_CONVERSIONS__', # Undefine the disable flag
            '-U__CUDA_NO_HALF2_OPERATORS__',     # Undefine the disable flag
            # generate code for H100 (compute_90) and fall back on A100 (compute_80)
            '-gencode', 'arch=compute_90,code=sm_90',
            '-gencode', 'arch=compute_80,code=sm_80',
            # keep your runtime link flags
            '-L/usr/local/cuda/lib64', '-lcudadevrt', '-lcudart'
        ]
    }

setup(
    name='clustencuda',
    version='0.1',
    author='Ziwen Chen',
    author_email='chenziw@oregonstate.edu',
    description='Cluster Attention CUDA Kernel',
    ext_modules=[
        CUDAExtension('clustenqk_cuda', [
            'clustenqk_cuda.cpp',
            'clustenqk_cuda_kernel.cu',
        ], include_dirs=include_dirs, extra_compile_args=extra_compile_args),
        CUDAExtension('clustenav_cuda', [
            'clustenav_cuda.cpp',
            'clustenav_cuda_kernel.cu',
        ], include_dirs=include_dirs, extra_compile_args=extra_compile_args),
        CUDAExtension('clustenwf_cuda', [
            'clustenwf_cuda.cpp',
            'clustenwf_cuda_kernel.cu',
        ], include_dirs=include_dirs, extra_compile_args=extra_compile_args),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
