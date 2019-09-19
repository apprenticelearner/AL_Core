#!/usr/bin/env python

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='apprentice',
    version=0.1,
    install_requires=requirements,
    packages=find_packages(),
    include_package_data=True,
    license='MIT',
   
)
