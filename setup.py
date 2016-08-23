#! /usr/bin/env python
# S.D.G

__author__ = 'Ben Johnston'
__revision__ = '0.2.2'
__date__ = 'Friday 19 August 00:13:47 AEST 2016'
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
