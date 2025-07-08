"""Visualization context and utility functions."""

import matplotlib.pyplot as plt
from typing import List


def get_color_palette() -> List[str]:
    """Get extended color palette for plotting."""
    # take matplotlib standard color wheel
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # add extra color palette
    colors += ["#FFA07A","#20B2AA","#FF6347","#808000","#FF00FF","#FFD700","#00FF00","#00FFFF","#0000FF","#8A2BE2"]
    return colors


class VisualizationContext:
    """Context object containing visualization data and settings."""
    
    def __init__(self):
        self.xtick_pos = None
        self.xtick_label = None
        self.real_los = None
        self.graph_colors = get_color_palette()
        self.xlims = None
        self.results_folder = None
        self.figures_folder = None
        self.animation_folder = None
