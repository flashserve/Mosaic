# file: setup.py
# Build script for flash_sample CUDA extensions
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='flash_sample',
    ext_modules=[
        CUDAExtension(
            'flash_sample_low_confidence',
            ['flash_sample_low_confidence.cu']
        ),
        CUDAExtension(
            'flash_sample_entropy',
            ['flash_sample_entropy.cu']
        ),
        CUDAExtension(
            'flash_sample_entropy_witht',
            ['flash_sample_entropy_witht.cu']
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

