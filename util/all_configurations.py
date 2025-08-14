# %%

# %load_ext autoreload
# %autoreload 2
import matplotlib.pyplot as plt

from los_estimator.config import (
    AnimationConfig,
    DataConfig,
    DebugConfig,
    ModelConfig,
    OutputFolderConfig,
    VisualizationConfig,
)
from los_estimator.core import *
from los_estimator.fitting.errors import ErrorType

# %%

data_config = DataConfig(
    base_path="C:/data/src/los-estimator/los-estimator/data",
    cases_file="./cases.csv",
    icu_occupancy_file="./Intensivregister_Bundeslaender_Kapazitaeten.csv",
    los_file="../01_create_los_profiles/berlin/output_los/los_berlin_all.csv",
    init_params_file="../02_fit_los_distributions/output_los/los_berlin_all/fit_results.csv",
    mutants_file="./VOC_VOI_Tabelle.xlsx",
    start_day="2021-07-29",
    end_day="2025-01-01",
)


model_config = ModelConfig(
    kernel_width=120,
    los_cutoff=60,  # Ca. 90% of all patients are discharged after 41 days
    smooth_data=False,
    train_width=42 + 60,
    test_width=21,  # 28 * 4
    step=7,
    error_fun=ErrorType.MSE,
    reuse_last_parametrization=True,
    variable_kernels=True,
    distributions=[
        # "lognorm",
        # "weibull",
        "gaussian",
        "exponential",
        # "gamma",
        # "beta",
        "cauchy",
        "t",
        # "invgauss",
        "linear",
        # "block",
        # "sentinel",
        "compartmental",
    ],
)

output_config = OutputFolderConfig("./results")

debug_configuration = DebugConfig(one_window=False, less_windows=True, less_distros=False, only_linear=False)

visualization_config = VisualizationConfig(
    save_figures=True,
    show_figures=True,
)
animation_config = AnimationConfig(debug_animation=False, debug_hide_failed=True, show_figures=True, save_figures=False)

estimator = LosEstimationRun(
    data_config, output_config, model_config, debug_configuration, visualization_config, animation_config
)
