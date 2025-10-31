# LOS Estimator

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Length of Stay Estimator for ICU data using deconvolution methods**

A Python package for estimating Length of Stay (LOS) distributions in Intensive Care Units (ICU) using advanced deconvolution techniques. This tool is particularly useful for healthcare researchers and data scientists working with COVID-19 hospitalization data and ICU capacity planning.

## 🚀 Features

- **Deconvolution Analysis**: Advanced mathematical techniques to estimate LOS distributions
- **Multiple Distribution Models**: Support for various probability distributions (Cauchy, exponential, compartmental)
- **ICU Data Processing**: Specialized tools for processing intensive care unit occupancy data
- **Visualization**: Comprehensive plotting and animation capabilities for data exploration
- **Configuration Management**: Flexible TOML-based configuration system
- **CLI Interface**: Command-line tool for batch processing and automation
- **Packaged Data**: Built-in access to German COVID-19 hospitalization datasets

## 📦 Installation

### From PyPI (Recommended)
```bash
pip install los-estimator
```

### From Source
```bash
git clone https://github.com/los-estimator/los-estimator.git
cd los-estimator
pip install -e .
```

### Development Installation
```bash
git clone https://github.com/los-estimator/los-estimator.git
cd los-estimator
pip install -e .[dev]
```

## 🎯 Quick Start

### Command Line Interface

```bash
# Run with default configuration
los-estimator

# Run with custom configuration
los-estimator --config_file path/to/config.toml

# Override specific parameters
los-estimator --overwrite_config_file path/to/overrides.toml --silent-plots

# Get help
los-estimator --help
```

### Python API

```python
from los_estimator import LosEstimationRun
from los_estimator.config import load_configurations
from los_estimator.data.loader import load_cases_data, load_hospital_data

# Load configuration
config = load_configurations("config.toml")

# Create estimator
estimator = LosEstimationRun(
    data_config=config["data_config"],
    output_config=config["output_config"],
    model_config=config["model_config"],
    debug_config=config["debug_config"],
    visualization_config=config["visualization_config"],
    animation_config=config["animation_config"],
)

# Run analysis
estimator.run_analysis()
```

### Loading Packaged Data

```python
from los_estimator.data.loader import load_cases_data, load_hospital_data

# Load built-in datasets
cases_df = load_cases_data()
hospital_df = load_hospital_data()

print(f"Cases data shape: {cases_df.shape}")
print(f"Hospital data shape: {hospital_df.shape}")
```

## 📊 Data Sources

The package includes several German COVID-19 datasets:

- **Cases Data**: `cases.csv` - Daily case numbers with reference dates
- **ICU Capacity**: `Intensivregister_Bundeslaender_Kapazitaeten.csv` - State-level ICU capacities
- **ICU Demographics**: `Intensivregister_Deutschland_Altersgruppen.csv` - Age group ICU data
- **VOC/VOI Data**: `VOC_VOI_Tabelle.xlsx` - Variants of Concern/Interest tracking
- **Sample LoS**: `sample_los.csv` - Sample for a LoS distribution

## ⚙️ Configuration

Create a configuration file (TOML format) to customize the analysis:

```toml
[data_config]
start_date = "2020-03-01"
end_date = "2023-12-31"

[model_config]
distribution_types = ["exponential", "cauchy", "compartmental"]
fit_method = "mse"
use_variable_kernels = true

[visualization_config]
show_figures = true
save_figures = true
figure_format = "png"

[output_config]
output_directory = "results"
save_fit_results = true
save_configurations = true
```

## 📈 Example Workflows

### 1. Basic LOS Estimation

```python
from los_estimator import LosEstimationRun
from los_estimator.config import load_configurations

# Load default configuration
config = load_configurations()

# Modify parameters
config["data_config"]["start_date"] = "2020-06-01"
config["model_config"]["distribution_types"] = ["exponential"]

# Run estimation
estimator = LosEstimationRun(**config)
results = estimator.run_analysis()
```

### 2. Batch Processing Multiple Configurations

```python
import itertools
from los_estimator import LosEstimationRun

# Define parameter grid
distributions = [["exponential"], ["cauchy"], ["compartmental"]]
fit_methods = ["mse", "mae"]

# Run grid search
for dist, method in itertools.product(distributions, fit_methods):
    config = load_configurations()
    config["model_config"]["distribution_types"] = dist
    config["model_config"]["fit_method"] = method
    
    estimator = LosEstimationRun(**config)
    results = estimator.run_analysis()
```

### 3. Custom Data Analysis

```python
from los_estimator.data import DataLoader
import pandas as pd

# Load your own data
loader = DataLoader(config["data_config"])
custom_data = pd.read_csv("your_data.csv")

# Process and analyze
processed_data = loader.preprocess_data(custom_data)
# ... continue with analysis
```

## 🔬 Methods and Algorithms

### Deconvolution Techniques
- **Mathematical Foundation**: Based on inverse problem solving for LOS distribution estimation
- **Regularization**: Advanced techniques to handle ill-posed inverse problems
- **Multiple Kernels**: Support for various kernel functions in deconvolution

### Distribution Models
- **Exponential**: Simple exponential decay model
- **Cauchy**: Heavy-tailed distribution for outlier-robust estimation
- **Compartmental**: Multi-compartment models for complex LOS patterns

### Fitting Methods
- **MSE**: Mean Squared Error minimization
- **MAE**: Mean Absolute Error minimization
- **Custom Metrics**: Extensible framework for custom fitting criteria

## 📁 Project Structure

```
los-estimator/
├── los_estimator/           # Main package
│   ├── cli/                # Command-line interface
│   ├── config/             # Configuration management
│   ├── core/               # Core algorithms
│   ├── data/               # Data loading and processing
│   ├── evaluation/         # Model evaluation tools
│   ├── fitting/            # Distribution fitting
│   └── visualization/      # Plotting and animation
├── data/                   # Packaged datasets
├── examples/               # Example notebooks and scripts
├── tests/                  # Test suite
├── docs/                   # Documentation
└── results/                # Output directory (created at runtime)
```

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install -e .[test]

# Run all tests
pytest

# Run with coverage
pytest --cov=los_estimator

# Run specific test
pytest tests/test_data_loader.py
```

## 📚 Documentation

- **API Documentation**: [Link to docs]
- **Tutorials**: See `examples/` directory
- **Method Details**: [Link to methodology paper]
- **Configuration Reference**: [Link to config docs]

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/los-estimator/los-estimator.git
cd los-estimator

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### Code Style

We use several tools to maintain code quality:

```bash
# Format code
black los_estimator/

# Sort imports
isort los_estimator/

# Lint code
flake8 los_estimator/

# Type checking
mypy los_estimator/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

- **Authors**: LOS Estimator Team
- **Email**: los-estimator@example.com
- **GitHub**: https://github.com/los-estimator/los-estimator
- **Issues**: https://github.com/los-estimator/los-estimator/issues

## 🙏 Acknowledgments

- Data sources: Robert Koch Institute (RKI), German Intensive Care Registry
- Methodology: Based on deconvolution techniques for epidemiological data
- Inspiration: COVID-19 pandemic response and ICU capacity planning research

## 📊 Citation

If you use this package in your research, please cite:

```bibtex
@software{los_estimator,
  author = {LOS Estimator Team},
  title = {LOS Estimator: Length of Stay Estimation for ICU Data},
  url = {https://github.com/los-estimator/los-estimator},
  version = {1.1.0},
  year = {2025}
}
```

## 🗓️ Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

---

**Made with ❤️ for healthcare research and ICU capacity planning**
