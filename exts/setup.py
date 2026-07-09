from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='BpOps',
    ext_modules=[
        CUDAExtension(
            name='BpOps',
            sources=['bp_cuda.cpp', 'bp_cuda_kernel.cu'],
            extra_compile_args={
                'cxx': ['-O3', '-std=c++17'],
                'nvcc': [
                    '-O3',
                    '-std=c++17',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    '-gencode=arch=compute_89,code=sm_89',  # You can change based on your GPU
                    '-ccbin=/usr/bin/g++'                   # Ensures nvcc uses system g++ not conda-cc
                ]
            }
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
