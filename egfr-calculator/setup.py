"""Setup script for the eGFR package."""

from setuptools import setup, find_packages

import os

if os.path.exists("README.md"):
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = (
        "A clinically-validated eGFR library that calculates kidney function "
        "from serum creatinine, cystatin C, and demographic factors."
    )

with open("requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

# Separate test dependencies from install dependencies
test_requires = [dep for dep in install_requires if dep.startswith("pytest")]
install_requires = [dep for dep in install_requires if not dep.startswith("pytest")]

setup(
    name="eGFR",
    version="0.1.0",
    author="Outlabs",
    description=(
        "A clinically-validated eGFR library that calculates kidney function "
        "from serum creatinine, cystatin C, and demographic factors."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks"]),
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require={
        "test": test_requires,
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
