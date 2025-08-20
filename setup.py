"""Setup configuration for LOS Estimator package."""

from pathlib import Path

from setuptools import find_packages, setup

# Read the contents of README file
this_directory = Path(__file__).parent
try:
    long_description = (this_directory / "README.md").read_text(encoding="utf-8")
except FileNotFoundError:
    long_description = (
        "Length of Stay Estimator for ICU data using deconvolution methods"
    )

setup(
    name="los_estimator",
    version="1.1.0",
    author="LOS Estimator Team",
    author_email="los_estimator@example.com",
    description="Length of Stay Estimator for ICU data using deconvolution methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/los_estimator/los_estimator",
    project_urls={
        "Bug Reports": "https://github.com/los_estimator/los_estimator/issues",
        "Source": "https://github.com/los_estimator/los_estimator",
        "Documentation": "https://los_estimator.readthedocs.io/",
    },
    packages=find_packages(exclude=["tests*", "docs*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="healthcare, ICU, length-of-stay, deconvolution, medical-data",
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "tqdm>=4.65.0",
        "numba>=0.58.0",
        "openpyxl>=3.1.0",  # For Excel file reading
        "toml>=0.10.2",  # For configuration files
        "click>=8.1.0",  # For CLI interface
        "dill",  # TODO version
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
            "pre-commit>=3.3.0",
            "jupyter>=1.0.0",
            "ipython>=8.14.0",
            "jupyterlab>=4.0.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "nbsphinx>=0.9.0",
            "sphinx-autodoc-typehints>=1.24.0",
            "myst-parser>=2.0.0",
        ],
        "test": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "pytest-xdist>=3.3.0",
        ],
        "lint": [
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
            "pre-commit>=3.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "los_estimator=los_estimator.cli:main",
        ],
    },
    package_data={
        "los_estimator": [
            "config/*.toml",
            "default_config.toml",
        ],
        "": [
            "data/*.csv",
            "data/*.xlsx",
            "data/dynamic/*.csv",
            "data/prepare_hospitalizations/*.csv",
        ],
    },
    data_files=[
        (
            "los_estimator/data",
            [
                "data/cases.csv",
                "data/hosp_ag.csv",
                "data/Intensivregister_Bundeslaender_Kapazitaeten.csv",
                "data/Intensivregister_Deutschland_Altersgruppen.csv",
                "data/VOC_VOI_Tabelle.xlsx",
            ],
        ),
        (
            "los_estimator/data/dynamic",
            [
                "data/dynamic/los_berlin_all.csv",
                "data/dynamic/los_berlin_fit_result.csv",
            ],
        ),
        (
            "los_estimator/data/prepare_hospitalizations",
            [
                "data/prepare_hospitalizations/Aktuell_Deutschland_COVID-19-Hospitalisierungen.csv",
            ],
        ),
    ],
    include_package_data=True,
    zip_safe=False,
)
