"""Model configuration for LOS Estimator."""

import os
import types
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List, Optional, Tuple

import toml

__all__ = [
    "ModelConfig",
    "DataConfig",
    "DebugConfig",
    "OutputFolderConfig",
    "AnimationConfig",
    "VisualizationConfig",
    "VisualizationContext",
    "load_configurations",
    "save_configurations",
    "default_config_path",
]

default_config_path = Path(__file__).parent.parent / "default_config.toml"

configuration_type = {}


def config(name=None):
    """Decorator to add a 'config_name' attribute to a dataclass.

    This decorator automatically converts a class to a dataclass and adds
    configuration metadata for serialization and identification purposes.

    Args:
        name (str, optional): Name for the configuration. If None, uses class name.

    Returns:
        callable: Decorator function that modifies the class.
    """

    def wrapper(cls):
        cls = dataclass(cls)
        _name = name if name else cls.__name__
        cls.config_name = _name
        configuration_type[_name] = cls
        return cls

    return wrapper


@config("model_config")
class ModelConfig:
    """Configuration for model fitting parameters.

    Contains all parameters related to the length of stay estimation model
    including kernel settings, data windowing, and optimization options.

    Attributes:
        kernel_width (int): Width of the distribution kernel in days.
        los_cutoff (int): Number of initial days to exclude from fitting.
        smooth_data (bool): Whether to apply smoothing to input data.
        train_width (int): Width of training window in days.
        test_width (int): Width of test window in days.
        step (int): Step size for sliding window analysis.
        error_fun (str): Error function to use for optimization.
    """

    kernel_width: int = 120
    los_cutoff: int = 60
    smooth_data: bool = False
    train_width: int = 42 + 60
    test_width: int = 21
    step: int = 7
    error_fun: str = "mse"
    reuse_last_parametrization: bool = True
    variable_kernels: bool = True
    distributions: List[str] = field(
        default_factory=lambda: [
            "lognorm",
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
        ]
    )
    run_name: str = ""
    ideas = types.SimpleNamespace(
        los_change_penalty=["..."],
        fitting_err=[
            "mse",
            "mae",
            "rel_err",
            "weighted_mse",
            "inv_rel_err",
            "capacity_err",
            "...",
        ],
        presenting_err=["..."],
    )


@config("data_config")
class DataConfig:
    """Configuration for data loading and processing.

    Specifies file paths and date ranges for loading hospital data
    required for length of stay analysis.

    Attributes:
        cases_file (str): Path to file containing case/admission data.
        icu_occupancy_file (str): Path to ICU occupancy time series data.
        los_file (str): Path to length of stay reference data.
        init_params_file (str): Path to initial parameter configuration.
        start_day (str): Start date for analysis period.
        end_day (str): End date for analysis period.
    """

    cases_file: str
    icu_occupancy_file: str
    los_file: str
    init_params_file: str
    start_day: str
    end_day: str

    mutants_file: Optional[str] = None

    def __post_init__(self):
        pass


@config("debug_config")
class DebugConfig:
    """Configuration for debugging and development options.

    Contains flags to enable various debugging modes that can speed up
    development by reducing the scope of analysis.

    Attributes:
        one_window (bool): Process only one time window for quick testing.
        less_windows (bool): Process fewer time windows than normal.
        less_distros (bool): Test with fewer distribution types.
        only_linear (bool): Use only linear distribution for testing.
    """

    one_window: bool = False
    less_windows: bool = False
    less_distros: bool = False
    only_linear: bool = False


@config("output_config")
class OutputFolderConfig:
    """Configuration for output folder structure.

    Manages the organization of output files including results, figures,
    animations, and metrics in a structured directory hierarchy.

    Attributes:
        base (str): Base directory for all outputs.
        run_name (str): Name of the specific analysis run.
    """

    base: str
    run_name: str

    def build(self):
        """Build the output directory structure.

        Creates the full paths for results, figures, animation, and metrics
        subdirectories based on the base path and run name.
        """
        if not self.run_name:
            return
        self.results = os.path.join(self.base, self.run_name)
        self.figures = os.path.join(self.results, "figures")
        self.animation = os.path.join(self.results, "animation")
        self.metrics = os.path.join(self.results, "metrics")

    def __post_init__(self):
        self.build()


@config("animation_config")
class AnimationConfig:
    """Configuration for animation generation and display.

    Controls how animations are created and displayed, including naming
    conventions and debugging options for animation generation.

    Attributes:
        show_figures (bool): Whether to display animations when created.
        save_figures (bool): Whether to save animation files to disk.
        debug_animation (bool): Enable debugging mode for animations.
        debug_hide_failed (bool): Hide failed fits in debug animations.
        alternative_names (List[Tuple[str, str]]): Alternative display names for models.
        replace_short_names (List[Tuple[str, str]]): Short name replacements for displays.
    """

    show_figures: bool = False
    save_figures: bool = True
    debug_animation: bool = False
    debug_hide_failed: bool = False
    alternative_names: List[Tuple[str, str]] = field(
        default_factory=lambda: [
            ("block", "Constant Discharge"),
            ("sentinel", "Baseline: Sentinel"),
        ]
    )
    replace_short_names: List[Tuple[str, str]] = field(
        default_factory=lambda: [
            ("exponential", "exp"),
            ("gaussian", "gauss"),
            ("compartmental", "comp"),
        ]
    )
    distro_colors: dict[str, str] = field(default_factory=lambda: {})
    distro_patches: dict[str, str] = field(default_factory=lambda: {})


@config("visualization_config")
class VisualizationConfig:
    """Configuration for output and visualization.

    Controls all aspects of plot generation including figure size, styling,
    colors, and output quality settings.

    Attributes:
        save_figures (bool): Whether to save plots to files.
        show_figures (bool): Whether to display plots on screen.
        xlims (Tuple[int, int]): X-axis limits for time series plots.
        figsize (Tuple[int, int]): Default figure size in inches.
        style (str): Matplotlib style to use for plots.
        colors (List[str]): Custom color palette for plots.
        savefig_facecolor (str): Background color for saved figures.
        savefig_dpi (int): DPI for saved figure files.
        figure_dpi (int): DPI for displayed figures.
    """

    save_figures: bool = True
    show_figures: bool = True

    xlims: Tuple[int, int] = (-30, 725)
    figsize: Tuple[int, int] = (12, 8)
    style: str = "seaborn-v0_8"
    colors: List[str] = field(default_factory=lambda: [])
    savefig_facecolor = "white"
    savefig_dpi: int = 300
    figure_dpi: int = 100


@dataclass
class VisualizationContext:
    """Runtime context for visualization components.

    Stores runtime information needed for generating consistent visualizations
    across different components, including axis formatting and folder paths.

    Attributes:
        xtick_pos (Tuple): X-axis tick positions for time series plots.
        xtick_label (Tuple): X-axis tick labels for time series plots.
        real_los (Tuple): Real length of stay data for reference.
        xlims (Tuple): X-axis limits for plots.
        results_folder (str): Path to results output folder.
        figures_folder (str): Path to figures output folder.
        animation_folder (str): Path to animation output folder.
    """

    xtick_pos: Tuple = ()
    xtick_label: Tuple = ()
    real_los: Tuple = ()
    xlims: Tuple = (-30, 725)
    results_folder: str = ""
    figures_folder: str = ""
    animation_folder: str = ""


def dict_to_config(config_dict, config_class):
    """Convert a dictionary to a configuration object.

    Filters the dictionary to only include fields that exist in the target
    configuration class, then creates an instance with those values.

    Args:
        config_dict (dict): Dictionary with configuration values.
        config_class (type): Configuration class to instantiate.

    Returns:
        object: Instance of config_class with values from config_dict.
    """
    field_names = {field.name for field in fields(config_class)}
    filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}
    return config_class(**filtered_dict)


def load_configurations(path):
    """Load configurations from a TOML file.

    Reads a TOML configuration file and converts it to appropriate
    configuration objects based on registered configuration types.

    Args:
        path (str or Path): Path to the TOML configuration file.

    Returns:
        dict: Dictionary mapping configuration names to configuration objects.
    """
    with open(path, "r") as f:
        loaded_config = toml.load(f)

    configs = {}
    for name in loaded_config.keys():

        if name not in configuration_type:
            continue

        configs[name] = dict_to_config(loaded_config[name], configuration_type[name])
    return configs


def save_configurations(path, configurations):
    """Save configurations to a TOML file.

    Converts configuration objects to dictionaries and saves them
    as a TOML file for future loading.

    Args:
        path (str or Path): Path where to save the TOML file.
        configurations (list): List of configuration objects to save.
    """
    config_dicts = {config.config_name: asdict(config) for config in configurations}
    with open(path, "w") as f:
        toml.dump(config_dicts, f)
