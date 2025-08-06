"""Model configuration for LOS Estimator."""
import os
from dataclasses import dataclass
from typing import List
import types


from typing import Optional

configuration_type = {}

def config(name=None):
    """Decorator to add a 'config_name' attribute to a dataclass."""
    def wrapper(cls):
        cls = dataclass(cls)        
        _name = name if name else cls.__name__
        cls.config_name = _name
        configuration_type[_name] = cls
        return cls
    return wrapper

@config("model_config")
class ModelConfig:
    kernel_width: int = 120
    los_cutoff: int = 60
    smooth_data: bool = False
    train_width: int = 42 + 60
    test_width: int = 21
    step: int = 7
    error_fun: str = "mse"
    reuse_last_parametrization: bool = True
    variable_kernels: bool = True
    distributions: list[str] = (
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
    )
    ideas = types.SimpleNamespace(
        los_change_penalty=["..."],
        fitting_err=["mse", "mae", "rel_err", "weighted_mse", "inv_rel_err", "capacity_err", "..."],
        presenting_err=["..."]
    )


@config("data_config")
class DataConfig:
    """Configuration for data loading and processing."""
    cases_file: str 
    icu_occupancy_file: str 
    los_file: str 
    init_params_file: str 
    start_day: str 
    end_day: str 
    
    base_path: str  ="."
    mutants_file: Optional[str] = None
    
    def get_cases_file(self):
        """Return the path to the cases file."""
        return os.path.join(self.base_path, self.cases_file)

    def get_icu_occupancy_file(self):
        """Return the path to the ICU occupancy file."""
        return os.path.join(self.base_path, self.icu_occupancy_file)

    def get_los_file(self):
        """Return the path to the LOS file."""
        return os.path.join(self.base_path, self.los_file)

    def get_init_params_file(self):
        """Return the path to the initial parameters file."""
        return os.path.join(self.base_path, self.init_params_file)

    def get_mutants_file(self):
        """Return the path to the mutants file."""
        if self.mutants_file is None:
            return None
        return os.path.join(self.base_path, self.mutants_file)
    
    def __post_init__(self):
        pass

@config("debug_config")
class DebugConfig:
    one_window: bool = False
    less_windows: bool = False
    less_distros: bool = False
    only_linear: bool = False
        

@config("output_config")
class OutputFolderConfig:
    base: str
    run_name: str = None
    

    def build(self):
        if self.run_name is None:
            return
        self.results = os.path.join(self.base, self.run_name)
        self.figures = os.path.join(self.results, "figures")
        self.animation = os.path.join(self.results, "animation")

    def __post_init__(self):
        self.build()



    
@config("animation_config")
class AnimationConfig:
    show_figures: bool = False
    save_figures: bool = True
    debug_animation: bool = False
    debug_hide_failed: bool = False
    alternative_names: list[tuple[str,str]] = (("block", "Constant Discharge"), ("sentinel", "Baseline: Sentinel"))
    replace_short_names: list[tuple[str,str]] = (("exponential", "exp"), ("gaussian", "gauss"), ("compartmental", "comp"))
    distro_colors: dict[str,str] = None
    distro_patches: dict[str,str] = None


@config("visualization_config")
class VisualizationConfig:
    """Configuration for output and visualization."""
    save_figures: bool = True
    show_figures: bool = True

    xlims: tuple[int,int] = (-30, 725)
    figsize: tuple[int,int] = (12, 8)
    style: str = "seaborn-v0_8"
    colors: List[str] = None
    savefig_facecolor  = 'white'
    savefig_dpi:int  = 300
    figure_dpi:int  = 100
    
    
   
@dataclass
class VisualizationContext:
    xtick_pos: list = None
    xtick_label: list = None
    real_los: list = None
    xlims: tuple = (-30, 725)
    results_folder: str = ""
    figures_folder: str = ""
    animation_folder: str = ""
    
    