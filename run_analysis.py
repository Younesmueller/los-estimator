# %%

# %load_ext autoreload
# %autoreload 2
import os

import matplotlib.pyplot as plt
import numpy as np

from util.comparison_data_loader import load_comparison_data

print("Let's Go!")
# %%
less_windows = True
compare_all_fit_results = load_comparison_data(less_windows)
print("Comparison data loaded successfully.")


# %%
def _compare_all_fitresults(all_fit_results, compare_all_fit_results):
    print("Starting comparison of all fit results...")

    all_successful = True
    for distro in compare_all_fit_results.keys():
        if distro not in all_fit_results:
            print(f"❌ Distribution {distro} not found in comparison results.")
            all_successful = False
    for distro, fit_result in all_fit_results.items():
        if distro == "compartmental":
            continue
        if distro not in compare_all_fit_results:
            print(f"❌ Distribution {distro} not found in comparison results.")
            all_successful = False
            continue

        comp_fit_result = compare_all_fit_results[distro]
        if fit_result.all_kernels.shape != comp_fit_result.all_kernels.shape:
            print(f"❌ Shape mismatch for kernels in distribution: {distro}")
            print(f"Expected shape: {comp_fit_result.all_kernels.shape}, but got: {fit_result.all_kernels.shape}")
            all_successful = False
            continue

        if not np.allclose(fit_result.all_kernels, comp_fit_result.all_kernels, atol=1e-4):
            print(f"❌ Kernel comparison failed for distribution: {distro}")
            print(f"Kernel Difference: {np.abs(fit_result.all_kernels - comp_fit_result.all_kernels).max():.4f}")
            print("-" * 50)
            all_successful = False
            continue

        train_error_diff = np.abs(
            fit_result.train_relative_errors.mean() - comp_fit_result.train_relative_errors.mean()
        )
        test_error_diff = np.abs(fit_result.test_relative_errors.mean() - comp_fit_result.test_relative_errors.mean())

        if train_error_diff > 1e-4 or test_error_diff > 1e-4:
            print(f"❌ Comparison failed for distribution: {distro}")
            print(f"Train Error Difference: {train_error_diff:.4f}")
            print(f"Test Error Difference: {test_error_diff:.4f}")
            print("-" * 50)
            all_successful = False
        else:
            print(f"✅ Comparison successful for distribution: {distro}")

    if all_successful:
        print("✅ All distributions compared successfully!")
    else:
        print("❌ Some distributions failed the comparison.")
        return fit_result.train_relative_errors, comp_fit_result.train_relative_errors


# %%
from los_estimator.estimation_run import LosEstimationRun, load_configurations, default_config_path, load_configurations
from los_estimator.config import update_configurations

cfg = load_configurations(default_config_path)
overwrite_cfg = load_configurations(default_config_path.parent / "overwrite_config.toml")


model_config = cfg["model_config"]
data_config = cfg["data_config"]
output_config = cfg["output_config"]
debug_config = cfg["debug_config"]
visualization_config = cfg["visualization_config"]
animation_config = cfg["animation_config"]

visualization_config.show_figures = False
animation_config.show_figures = False


update_configurations(cfg, overwrite_cfg)


# def update(obj, **kwargs):
#     for key, value in kwargs.items():
#         setattr(obj, key, value)
#     return obj


# model_config = update(
#     model_config,
#     kernel_width=120,
#     los_cutoff=60,  # Ca. 90% of all patients are discharged after 41 days
#     smooth_data=False,
#     train_width=42 + 60,
#     test_width=21,  # 28 * 4
#     step=7,
#     error_fun="mse",
#     reuse_last_parametrization=True,
#     variable_kernels=True,
#     distributions=[
#         # "lognorm",
#         # "weibull",
#         "gaussian",
#         "exponential",
#         # "gamma",
#         # "beta",
#         "cauchy",
#         "t",
#         # "invgauss",
#         "linear",
#         # "block",
#         # "sentinel",
#         "compartmental",
#     ],
# )


# animation_config.debug_animation = True


# debug_config = update(
#     debug_config,
#     one_window=False,
#     less_windows=False,
#     less_distros=False,
#     only_linear=False,
# )

# visualization_config = update(visualization_config,
#     save_figures=True,
#     show_figures=True,
# )
# animation_config = update(animation_config,
#     debug_animation=False,
#     debug_hide_failed=True,
#     show_figures=True,
#     save_figures=False
# )

estimator = LosEstimationRun(
    data_config,
    output_config,
    model_config,
    debug_config,
    visualization_config,
    animation_config,
)


estimator.run_analysis(vis=False)

_compare_all_fitresults(estimator.all_fit_results, compare_all_fit_results)
