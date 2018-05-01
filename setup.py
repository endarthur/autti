#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages

setup(
    name="auttitude",
    version="0.1.0",
    packages=find_packages(),

    install_requires=[
        'numpy',
        'matplotlib',
    ],

    # metadata for upload to PyPI
    author="Arthur Endlein",
    author_email="endarthur@gmail.com",
    description="library for analysis of structural geology data",
    license="MIT",
    keywords="geology attitude stereonet projection structural",
    url="https://github.com/endarthur/autti",
    dowload_url="https://github.com/endarthur/autti/archive/v0.1.0.tar.gz",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
    ],
)