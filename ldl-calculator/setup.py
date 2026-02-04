"""Setup script for ldlC package.

LDL Cholesterol Estimation Model - A clinically-validated LDL cholesterol
estimation algorithm implementing multiple mechanistic equations (Friedewald,
Martin-Hopkins, Sampson) enhanced by a unified machine learning model.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements.txt
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, "r", encoding="utf-8") as f:
        install_requires = [
            line.strip() for line in f if line.strip() and not line.startswith("#")
        ]
else:
    install_requires = [
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "lightgbm",
        "matplotlib",
    ]

# Read long description from README if it exists
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = __doc__

setup(
    name="ldlC",
    version="0.1.0",
    description="LDL Cholesterol Estimation Model implementing Friedewald, Martin-Hopkins, Sampson equations and hybrid ML",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="OutLabs",
    python_requires=">=3.8",
    packages=find_packages(exclude=["tests", "tests.*", "notebooks"]),
    install_requires=install_requires,
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "ldl",
        "cholesterol",
        "lipid",
        "friedewald",
        "martin-hopkins",
        "sampson",
        "machine-learning",
        "clinical",
    ],
    entry_points={},
)
