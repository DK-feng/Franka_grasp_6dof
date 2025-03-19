from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fps_cuda',
    ext_modules=[
        CUDAExtension(
            'fps_cuda',
            [
                'fps_cuda.cpp',          # PyTorch 接口封装
                'fps_cuda_kernel.cu'     # 核心 CUDA 实现
            ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
