__author__ = 'Konstantin Weddige'
from setuptools import setup, Extension

setup(
    name='Miniball',
    version='0.2',
    description='Smallest Enclosing Balls of Points',
    author='Bernd Gärtner, Konstantin Weddige',
    packages=['miniball',],
    ext_modules=[
        Extension(
            'miniball.bindings',
            ['src/miniballmodule.cpp', ],
            language='c++',
        ),
    ],
)
