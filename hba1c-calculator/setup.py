"""Setup script for hba1cE package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements from requirements.txt
requirements_path = Path(__file__).parent / "requirements.txt"
install_requires = []
if requirements_path.exists():
    install_requires = [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

# Read long description from README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="hba1cE",
    version="0.1.0",
    description="HbA1c Estimation from Routine Blood Markers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Outlabs",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks"]),
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
