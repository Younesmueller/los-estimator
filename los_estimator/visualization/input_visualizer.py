"""Input data visualization components."""

import matplotlib.pyplot as plt

from los_estimator.data import DataPackage

from ..config import VisualizationContext
from .base import VisualizerBase


class InputDataVisualizer(VisualizerBase):
    """Visualizer for input data analysis.

    Provides methods to create plots and visualizations of the input data
    including ICU occupancy, admissions, and variant data.

    Attributes:
        vc (VisualizationContext): Context for consistent plot formatting.
        data (DataPackage): Data package containing all input data.
        save_figures (bool): Whether to save generated figures.
        show_figures (bool): Whether to display generated figures.
    """

    def __init__(
        self,
        visualization_config,
        visualization_context: VisualizationContext,
        data=None,
    ):
        """Initialize the input data visualizer.

        Args:
            visualization_config: Configuration for plot styling and output.
            visualization_context (VisualizationContext): Shared visualization context.
            data (DataPackage, optional): Input data to visualize. Defaults to None.
        """
        super().__init__(visualization_config)
        self.vc: VisualizationContext = visualization_context
        self.data: DataPackage = data
        self.save_figures = False
        self.show_figures = True

    def show_input_data(self):
        """Show overview of input data.

        Creates a comprehensive overview plot showing all major input
        data series including incidences and ICU occupancy.
        """
        self.data.df_occupancy.plot(subplots=True)
        plt.suptitle("Incidences and ICU Occupancy")
        self._show()

    def plot_icu_data(self):
        """Plot ICU-specific data.

        Creates detailed plots of ICU admissions and occupancy data
        with proper labeling and formatting for analysis.
        """
        fig, ax = self._get_subplots(2, 1, figsize=(10, 5), sharex=True)

        self.data.df_occupancy["new_icu_smooth"].plot(ax=ax[1], label="new_icu", color="orange")
        self.data.df_occupancy["icu"].plot(ax=ax[0], label="AnzahlFall")
        ax[0].set_title("Tägliche Neuzugänge ICU, geglättet")
        ax[1].set_title("ICU Bettenbelegung")
        plt.tight_layout()
        self._show()

    def plot_mutant_data(self):
        """Plot mutant/variant data.

        Visualizes variant or mutant strain data if available,
        showing temporal patterns and prevalence.
        """
        self.data.df_mutant.plot()
        plt.show()
