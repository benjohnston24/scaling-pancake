#! /usr/bin/env python
# S.D.G

__author__ = 'Ben Johnston'
__revision__ = '0.3.1'
__date__ = 'Wednesday 24 August 14:38:47 AEST 2016'
__license__ = 'MPL v2.0'

from setuptools import setup, find_packages

setup(
    name='nnet',
    description='neural net toolkit',
    url='',
    author='Ben Johnston',
    author_email='bjohnston24@gmail.com',
    version=__revision__,
    packages=find_packages(),
    #packages=open('requirements.txt').read().split('\n')[:-1],
    #license=open('LICENSE.txt').read(),
    long_description=open('README.md').read(),
    )
