from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'binaural_dsp',
        ['src/binaural_dsp.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['/O2', '/std:c++17']
    ),
]

setup(
    name='binaural_dsp',
    ext_modules=ext_modules,
)
