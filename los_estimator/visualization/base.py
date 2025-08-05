"""Base visualizer class with common functionality."""

from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
from ..config import VisualizationConfig

def get_color_palette() -> List[str]:
    """Get extended color palette for plotting."""
    # take matplotlib standard color wheel
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # add extra color palette
    colors += ["#FFA07A","#20B2AA","#FF6347","#808000","#FF00FF","#FFD700","#00FF00","#00FFFF","#0000FF","#8A2BE2"]
    return colors




class VisualizerBase:
    """Base class for all visualizers."""

    def __init__(self, visualization_config: VisualizationConfig):
        self.visualization_config = visualization_config
        try:
            plt.style.use(visualization_config.style)
        except OSError:
            plt.style.use("default")
        
        self.figsize = visualization_config.figsize        
        self.colors = visualization_config.colors
        
        # Set high-quality defaults
        plt.rcParams['savefig.facecolor'] = visualization_config.savefig_facecolor
        plt.rcParams['savefig.dpi'] = visualization_config.savefig_dpi
        plt.rcParams['figure.dpi'] = 100

    def _figure(self, *args, **kwargs) -> plt.Figure:
        """Create a new figure with specified size and DPI."""
        figsize = kwargs.pop('figsize', self.figsize)
        return plt.figure(*args, figsize=figsize, **kwargs)

    def _get_subplots(self, *args, **kwargs) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Create subplots with specified number of rows and columns."""
        figsize = kwargs.pop('figsize', self.figsize)        
        return plt.subplots(*args, figsize=figsize, **kwargs)

    def _show(self, filename: str = None, fig: Optional[plt.Figure] = None):
        """Save the figure and show it."""
        if fig is None:
            fig = plt.gcf()

        if self.visualization_config.save_figures:
            if filename:
                if not filename.endswith('.png'):
                    filename = filename + '.png'
                full_path = self.visualization_config.figures_folder / filename
                fig.savefig(full_path, bbox_inches='tight')

        if self.visualization_config.show_figures:
            plt.show(block=False)
            plt.pause(0.001)
            plt.show()
        else:
            plt.clf()

    def _set_title(self, title: str, *args, **kwargs):
        """Set the title of the current figure."""
        run_name = self.model_config.run_name
        plt.title(title + "\n" + run_name, *args, **kwargs)
