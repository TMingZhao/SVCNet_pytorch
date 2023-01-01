from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
# os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"


setup(
    name='sampling',
    ext_modules=[
        CUDAExtension('sampling', [
            'sampling.cpp',
            'sampling_cuda.cu',],
        # extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']},
        include_dirs=["."])
    ],

    cmdclass={
        'build_ext': BuildExtension
    })
