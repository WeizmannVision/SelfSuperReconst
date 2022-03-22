#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='self_super_reconst',
    version='0.1.0',
    author="Guy Gaziv",
    author_email='guy.gaziv@weizmann.ac.il',
    description="Self-Supervised RGBD Reconstruction From Brain Activity (Official PyTorch implementation)",
    long_description=open("README.md", "r", encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    keywords="Self-Supervised Learning; Decoding; Encoding; fMRI Image Reconstruction; Classification; Vision; Depth Estimation",
    url='https://github.com/WeizmannVision/SelfSuperReconst',
    packages=find_packages(),
    include_package_data=True,
    tests_require=['pytest'],
    license="Yeda",
    classifiers=[
          'Intended Audience :: Science/Research',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    test_suite='tests',
)
