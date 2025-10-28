# Draft for Publication in SoftwareX

## 1. Motivation and Significance
The COVID-19 pandemic and other global health crises have highlighted the critical need for robust tools to predict ICU occupancy. Accurate predictions enable healthcare systems to allocate resources effectively and respond proactively to surges in demand. However, real patient data for Length of Stay (LoS) estimation is often difficult to access due to privacy concerns and limited availability.

This software addresses the challenge of retroactively estimating LoS distributions by framing the problem as a deconvolution between ICU admissions and occupancy rates. Using publicly available ICU data from Germany, the software provides a robust methodology for estimating LoS distributions, which are modeled as Probability Density Functions (PDFs) with a reduced parameter set. This approach reduces the complexity of the problem while maintaining accuracy.

The software also accounts for the dynamic nature of pandemics by enabling rolling fitting of kernels over time. This feature is particularly useful for adapting to changes in patient flow and disease progression during different phases of a pandemic.

## 2. Software Description

### Architecture
The software is designed as a modular Python package, leveraging `numpy` and `numba` for efficient computation. The key submodules include:

1. **`data`**: Handles data loading and preprocessing, ensuring compatibility with publicly available ICU datasets. It includes utilities for cleaning, normalizing, and transforming raw data into a format suitable for analysis.
2. **`fitting`**: Implements algorithms for fitting LoS distributions, including support for multiple probability distribution functions. The fitting process is optimized for performance and accuracy, leveraging advanced numerical techniques.
3. **`evaluation`**: Provides tools for computing error metrics to validate the accuracy of the fitted models. Metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE) are included to assess model performance.
4. **`visualization`**: Generates comparison plots and animations to visualize the fitting process and results. The module supports both static and dynamic visualizations, making it easier to interpret the outcomes.
5. **`estimation_run`**: Integrates all components to provide an end-to-end solution for LoS estimation. This module acts as the main entry point for users, orchestrating the entire workflow.

### Functionalities
- **Data Preprocessing**: Reads and preprocesses ICU data in formats commonly used in public datasets. This includes handling missing values, smoothing noisy data, and aligning time series.
- **Probability Distribution Functions**: Supports various PDFs, such as Gaussian, Exponential, and Weibull, for modeling LoS distributions. Users can select the most appropriate distribution based on their data characteristics.
- **Rolling Window Analysis**: Enables dynamic analysis by fitting kernels over rolling time windows. This feature is particularly useful for capturing temporal variations in LoS distributions.
- **Configuration-Driven Execution**: Allows users to customize and reproduce analyses through configuration files. The configuration system supports both default and user-defined settings, ensuring flexibility and reproducibility.

### Visualization
The software includes robust visualization tools to aid in interpreting results. Key features include:
- **Comparison Plots**: Evaluate the fit between observed and estimated data, providing insights into model accuracy.
- **Animated Visualizations**: Illustrate the evolution of LoS distributions over time, offering a dynamic view of changes in patient flow and occupancy patterns.
- **Customizable Outputs**: Users can customize plot styles, colors, and labels to suit their specific needs.

## 3. Illustrative Example
To demonstrate the software's capabilities, consider the following example:

1. **Data Loading**: The user loads ICU admission and occupancy data using the `data` module. For example, the `load_data` function can read CSV files and return cleaned data ready for analysis.
2. **Configuration**: A configuration file is prepared to specify the analysis parameters, such as the rolling window size, the type of PDF to use, and the output directory for results. The configuration can be written in TOML or JSON format.
3. **Estimation Run**: The `estimation_run` module is executed to perform the LoS estimation. This involves calling the `run_estimation` function, which integrates data preprocessing, fitting, and evaluation steps.
4. **Visualization**: The results are visualized using the `visualization` module. Static plots can be generated using the `plot_comparison` function, while animations can be created with the `animate_evolution` function.

This example highlights the software's ease of use and its ability to produce actionable insights from sparse and noisy data. The modular design ensures that each step can be customized or replaced as needed.

## 4. Impact
The software has significant implications for healthcare resource planning and pandemic preparedness. By providing accurate and robust LoS estimates, it enables healthcare systems to:
- **Optimize ICU Resource Allocation**: Ensure that beds, staff, and equipment are available where they are most needed.
- **Anticipate Bottlenecks**: Identify potential bottlenecks in patient care and take proactive measures to address them.
- **Adapt to Changing Conditions**: Use rolling analysis to respond to shifts in patient flow and disease progression during a pandemic.

As an open-source tool, the software also fosters collaboration and reproducibility. Researchers and practitioners can build upon its capabilities, contributing to a growing ecosystem of tools for healthcare analytics.

## 5. Conclusions and Future Work
In conclusion, this software represents a valuable contribution to the field of healthcare analytics. Its modular design, robust functionalities, and focus on visualization make it a powerful tool for LoS estimation.

Future work will focus on:
- **Enhancing Error Handling**: Improve error messages and logging to make the software more user-friendly.
- **Expanding Supported Distributions**: Add support for additional probability distribution functions, such as Log-Normal and Gamma.
- **Advanced Visualization Features**: Develop interactive dashboards for real-time data exploration and analysis.
- **Broader Dataset Compatibility**: Ensure compatibility with international datasets and different data formats.

## 6. Acknowledgments
We would like to thank all contributors to this project, as well as the organizations that provided the publicly available ICU data used in this study. Special thanks to the open-source community for their invaluable tools and libraries.

## 7. References
- Relevant literature on LoS estimation and deconvolution methods.
- Documentation for `numpy` and `numba` libraries.
- Publicly available ICU datasets used in the analysis.


