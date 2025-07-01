# LOS Estimator

A Python package for estimating length of stay (LOS) distributions in ICU settings using deconvolution techniques on time series data.

## Overview

The LOS Estimator package provides tools to:
- Load and preprocess ICU occupancy and admission data
- Fit various probability distributions to length of stay data
- Perform deconvolution analysis to estimate LOS distributions from time series
- Visualize results and compare different distribution fits

This package was developed to analyze COVID-19 ICU data in Germany, but can be adapted for other healthcare time series analysis.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd los-estimator

# Install the package
pip install -e .
```

## Quick Start

### Command Line Interface

```bash
# Run analysis with default parameters
los-estimator \
  --los-file data/los_distribution.csv \
  --incidence-file data/cases.csv \
  --icu-file data/icu_occupancy.csv \
  --init-params data/initial_params.csv \
  --mutants-file data/variants.xlsx

# Run with custom parameters
los-estimator \
  --los-file data/los_distribution.csv \
  --incidence-file data/cases.csv \
  --icu-file data/icu_occupancy.csv \
  --init-params data/initial_params.csv \
  --mutants-file data/variants.xlsx \
  --kernel-width 180 \
  --step 14 \
  --distributions lognorm weibull gamma
```

### Python API

```python
from los_estimator import LOSEstimator

# Initialize estimator
estimator = LOSEstimator()

# Run complete analysis
results = estimator.run_complete_analysis(
    los_file="data/los_distribution.csv",
    incidence_file="data/cases.csv", 
    icu_file="data/icu_occupancy.csv",
    init_params_file="data/initial_params.csv",
    mutants_file="data/variants.xlsx",
    kernel_width=120,
    distributions=['lognorm', 'weibull', 'gamma']
)

# Get best performing distribution
best_distro = estimator.get_best_distribution()
print(f"Best distribution: {best_distro}")

# Get summary statistics
summary = estimator.get_summary_statistics()
print(summary)
```

### Step-by-Step Analysis

```python
from los_estimator import LOSEstimator, EstimationParams

estimator = LOSEstimator()

# 1. Load data
data = estimator.load_data(
    los_file="data/los_distribution.csv",
    incidence_file="data/cases.csv",
    icu_file="data/icu_occupancy.csv", 
    init_params_file="data/initial_params.csv",
    mutants_file="data/variants.xlsx"
)

# 2. Configure parameters
params = estimator.configure_estimation(
    kernel_width=120,
    train_width=102,
    test_width=21,
    step=7,
    fit_admissions=True,
    error_function="mse"
)

# 3. Run estimation
results = estimator.run_estimation(['lognorm', 'weibull', 'gamma'])

# 4. Visualize results
saved_plots = estimator.visualize_results()
```

## Package Structure

```
los_estimator/
├── core/                 # Core estimation engine
│   ├── estimator.py     # Main LOSEstimator class
│   └── models.py        # Data structures and result classes
├── data/                 # Data loading utilities
│   └── loader.py        # DataLoader class
├── fitting/              # Fitting algorithms
│   ├── distributions.py # Distribution fitting
│   └── deconvolution.py # Deconvolution engine
├── visualization/        # Plotting and visualization
│   └── plots.py         # Visualizer class
└── utils/               # Utility functions
    └── helpers.py       # Helper functions
```

## Key Features

### Distributions Supported
- Log-normal
- Weibull  
- Gamma
- Exponential
- Normal
- Beta

### Error Functions
- Mean Squared Error (MSE)
- Weighted MSE
- Mean Absolute Error (MAE)

### Estimation Parameters
- `kernel_width`: Width of deconvolution kernel (default: 120)
- `train_width`: Training window width (default: 102)
- `test_width`: Test window width (default: 21)
- `step`: Step size between windows (default: 7)
- `fit_admissions`: Fit to admission vs incidence data (default: True)
- `smooth_data`: Use smoothed data (default: False)

### Visualization Options
- Time series overview plots
- Distribution comparison charts
- Fit quality heatmaps  
- Parameter evolution plots
- Prediction vs actual comparisons

## Data Requirements

The package expects the following input files:

1. **LOS Distribution** (CSV): Empirical length of stay distribution
2. **Incidence Data** (CSV): Daily case/incidence data with columns `AnzahlFall`, `daily`  
3. **ICU Occupancy** (CSV): ICU occupancy data with columns `datum`, `belegung_covid19`
4. **Initial Parameters** (CSV): Initial fitting parameters for distributions
5. **Variant Data** (Excel): Variant/mutant distribution over time

## Examples

See the `examples/` directory for detailed usage examples:
- `basic_usage.py`: Basic package usage
- Advanced analysis examples

## Requirements

- Python 3.8+
- NumPy
- Pandas  
- SciPy
- Matplotlib
- Seaborn
- Numba
- OpenPyXL (for Excel files)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.