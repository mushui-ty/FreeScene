#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name="freescene",
    packages=find_packages(exclude=["scripts"]),
)
