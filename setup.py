"""Setup configuration for LOS Estimator package."""

from pathlib import Path
from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.rst").read_text(encoding="utf-8")

setup(
    name="los_estimator",
    version="0.1",
    author_email="yomueller@ukaachen.de"
    author="Younes Müller",
    description="Length of Stay Estimator for ICU data using deconvolution methods",
    license="GPLv3",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=find_packages(exclude=["tests*", "docs*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "numba>=0.58.0",
        "openpyxl>=3.1.0",
        "toml>=0.10.2",
        "click>=8.1.0",
        "dill>=0.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": ["los_estimator=los_estimator.cli:main"],
    },
    package_data={
        "los_estimator": [
            "default_config.toml",
            "overwrite_config.toml",
            "data/**/*.csv",
            "data/**/*.xlsx",
        ],
    },
    include_package_data=True,
)
