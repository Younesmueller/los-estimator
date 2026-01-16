# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from generate_synthetic_data import generate_and_save_synthetic_data
from los_estimator.estimation_run import LosEstimationRun, load_configurations


# %%

kernel_width, data_path, kernel_path = generate_and_save_synthetic_data()


cfg = load_configurations("./synthetic_example.toml")

model_config = cfg["model_config"]
data_config = cfg["data_config"]
output_config = cfg["output_config"]
debug_config = cfg["debug_config"]
visualization_config = cfg["visualization_config"]
animation_config = cfg["animation_config"]

visualization_config.show_figures = True
animation_config.show_figures = True
data_config.icu_file = data_path
data_config.los_file = kernel_path
model_config.kernel_width = kernel_width

estimator = LosEstimationRun(
    data_config,
    output_config,
    model_config,
    debug_config,
    visualization_config,
    animation_config,
)
estimator.run_analysis()
