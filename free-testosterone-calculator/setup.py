"""
Setup configuration for the freeT package.

A clinically-validated free testosterone estimation algorithm from 
total testosterone (TT), SHBG, and albumin.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="freeT",
    version="0.1.0",
    author="Outlabs",
    author_email="",
    description="Free testosterone estimation using mechanistic solvers and ML models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nmillrr/outlabs",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "matplotlib",
    ],
    extras_require={
        "dev": [
            "pytest",
        ],
    },
)
