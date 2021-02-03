#!/usr/bin/env python3
"""Metadata for package to allow installation with pip."""

import os

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

# Use same version from code
# See 3 from
# https://packaging.python.org/guides/single-sourcing-package-version/
version = {}
with open(os.path.join("echonet", "__version__.py")) as f:
    exec(f.read(), version)

setuptools.setup(
    name="echonet-labs",
    description="Video-based AI for prediction of common lab values.",
    version=version["__version__"],
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "opencv-python",
        "scikit-image",
        "tqdm",
        "sklearn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ]
)
