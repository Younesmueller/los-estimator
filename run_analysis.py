# %%

# %load_ext autoreload
# %autoreload 2
import os

import matplotlib.pyplot as plt
import numpy as np

print("Let's Go!")

#

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


update_configurations(cfg, overwrite_cfg)

# %%
estimator = LosEstimationRun(
    data_config,
    output_config,
    model_config,
    debug_config,
    visualization_config,
    animation_config,
)
estimator.run_analysis()
# %%

estimator.set_up()
estimator.load_data()
estimator.fit()
estimator.evaluate()
estimator.save_results()
estimator.animate_results()

print("done.")


# %%

estimator2 = LosEstimationRun.load_run(os.path.dirname(estimator.output_config.model_data))

# %%
import dill
from pathlib import Path


def load_run(folder):
    path = Path(folder)

    cfg = load_configurations(path / "run_configurations.toml")
    model_config = cfg["model_config"]
    data_config = cfg["data_config"]
    output_config = cfg["output_config"]
    debug_config = cfg["debug_config"]
    visualization_config = cfg["visualization_config"]
    animation_config = cfg["animation_config"]

    run_nickname = None
    if "run_nickname" in cfg:
        run_nickname = cfg["run_nickname"]

    run = LosEstimationRun(
        data_config,
        output_config,
        model_config,
        debug_config,
        visualization_config,
        animation_config,
        run_nickname,
    )

    def _load(name):
        file = path / "model_data" / f"{name}.pkl"
        if file.exists():
            with open(file, "rb") as f:
                return dill.load(f)
        return None

    run.series_data = _load("series_data")
    run.all_fit_results = _load("all_fit_results")
    run.visualization_context = _load("visualization_context")

    return run


estimator2 = load_run(estimator.output_config.results)
estimator2.load_data()
# %%
estimator2.animate_results()
