# file: setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fused_ops',
    ext_modules=[
        CUDAExtension(
            'fused_ops',
            [
                # 现在只编译这一个统一的文件
                'flash_sample_confidence_unified.cu',
            ]
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)