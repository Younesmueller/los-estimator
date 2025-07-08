"""Base visualizer class with common functionality."""

import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
from .context import get_color_palette


class VisualizerBase:
    """Base class for all visualizers."""
    
    def __init__(self, style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use("default")
        
        self.figsize = figsize
        self.colors = get_color_palette()
        
        # Set high-quality defaults
        plt.rcParams['savefig.facecolor'] = 'white'
        plt.rcParams['savefig.dpi'] = 300
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

        if getattr(self, 'save_figs', False):
            if filename and not filename.endswith('.png'):
                filename = filename + '.png'
            if isinstance(getattr(self, 'figures_folder', ''), str):
                full_path = self.figures_folder + filename
            else:
                full_path = getattr(self, 'figures_folder', '') / filename
            fig.savefig(full_path, bbox_inches='tight')

        if getattr(self, 'show_figs', True):
            plt.show()
        else:
            plt.clf()

    def _set_title(self, title: str, *args, **kwargs):
        """Set the title of the current figure."""
        run_name = getattr(getattr(self, 'params', None), 'run_name', '')
        plt.title(title + "\n" + run_name, *args, **kwargs)
