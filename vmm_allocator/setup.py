#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Setup script for CUDA VMM Allocator PyTorch extension

import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 获取 CUDA 路径
cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
if cuda_home is None:
    import torch
    cuda_home = torch.utils.cpp_extension.CUDA_HOME

if cuda_home is None:
    print("Error: CUDA not found. Please set CUDA_HOME environment variable.")
    sys.exit(1)

print(f"Using CUDA from: {cuda_home}")

# 编译参数
extra_compile_args = {
    'cxx': [
        '-O3',
        '-std=c++17',
    ],
    'nvcc': [
        '-O3',
        '-std=c++17',
        '--expt-relaxed-constexpr',
    ]
}

# 源文件
sources = [
    'vmm_python_binding.cpp',
    'vmm_allocator.cpp',
]

# 包含目录
include_dirs = [
    os.path.join(cuda_home, 'include'),
    '.',
]

# 库目录和库
library_dirs = [os.path.join(cuda_home, 'lib64')]
libraries = ['cuda', 'cudart']

setup(
    name='vmm_allocator',
    version='0.1.0',
    author='vDLLM Team',
    description='CUDA Virtual Memory Management Allocator for PyTorch',
    ext_modules=[
        CUDAExtension(
            name='vmm_allocator',
            sources=sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    },
    python_requires='>=3.8',
)

