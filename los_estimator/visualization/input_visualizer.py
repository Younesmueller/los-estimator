"""Input data visualization components."""

import matplotlib.pyplot as plt

from los_estimator.data import DataPackage

from ..config import VisualizationContext
from .base import VisualizerBase


class InputDataVisualizer(VisualizerBase):
    """Visualizer for input data analysis."""

    def __init__(
        self,
        visualization_config,
        visualization_context: VisualizationContext,
        data=None,
    ):
        super().__init__(visualization_config)
        self.vc: VisualizationContext = visualization_context
        self.data: DataPackage = data
        self.save_figures = False
        self.show_figures = True

    def show_input_data(self):
        """Show overview of input data."""
        self.data.df_occupancy.plot(subplots=True)
        plt.suptitle("Incidences and ICU Occupancy")
        self._show()

    def plot_icu_data(self):
        """Plot ICU-specific data."""
        fig, ax = self._get_subplots(2, 1, figsize=(10, 5), sharex=True)

        self.data.df_occupancy["new_icu_smooth"].plot(
            ax=ax[1], label="new_icu", color="orange"
        )
        self.data.df_occupancy["icu"].plot(ax=ax[0], label="AnzahlFall")
        ax[0].set_title("Tägliche Neuzugänge ICU, geglättet")
        ax[1].set_title("ICU Bettenbelegung")
        plt.tight_layout()
        self._show()

    def plot_mutant_data(self):
        """Plot mutant/variant data."""
        self.data.df_mutant.plot()
        plt.show()
