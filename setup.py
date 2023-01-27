# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.

from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from torch.utils import cpp_extension

setup(name='dpf_cpp',
      ext_modules=[
          CUDAExtension('dpf_cpp', sources=[
              'dpf_wrapper.cu',
          ], extra_compile_args=['-std=c++17'],
          )
      ],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
)


