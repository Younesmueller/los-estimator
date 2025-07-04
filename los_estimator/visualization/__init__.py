#%%
# reload imports
import numpy as np
import types
from typing import List, Optional, Tuple, Union

import seaborn as sns

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from los_estimator.core import *


def get_color_palette():
    # take matplotlib standart color wheel
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # add extra color palette
    colors += ["#FFA07A","#20B2AA","#FF6347","#808000","#FF00FF","#FFD700","#00FF00","#00FFFF","#0000FF","#8A2BE2"]
    return colors

class VisualizerBase():
    def __init__(self,style: str = "seaborn-v0_8", figsize: Tuple[int, int] = (12, 8)):
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


    def _figure(self,*args,**kwargs) -> plt.Figure:
        """Create a new figure with specified size and DPI."""
        figsize = kwargs.pop('figsize', self.figsize)
        return plt.figure(*args,figsize=figsize,**kwargs)

    def _get_subplots(self, *args, **kwargs) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Create subplots with specified number of rows and columns."""
        figsize = kwargs.pop('figsize', self.figsize)        
        return plt.subplots(*args, figsize=figsize, **kwargs)


    def _show(self,filename: str=None, fig: Optional[plt.Figure] = None):
        """Save the figure and show it."""
        if fig is None:
            fig = plt.gcf()

        if self.save_figs:
            if not filename.endswith('.png'):
                filename = filename + '.png'
            if isinstance(self.figures_folder, str):
                full_path = self.figures_folder + filename
            else:
                full_path = self.figures_folder / filename
            fig.savefig(full_path, bbox_inches='tight')

        if self.show_figs:
            plt.show()
        else:
            plt.clf()

    def _set_title(self, title: str,*args,**kwargs):
        """Set the title of the current figure."""
        plt.title(title+"\n" + self.params.run_name, *args, **kwargs)

class InputDataVisualizer(VisualizerBase):
    def __init__(self, visualization_context:VisualizationContext,data=None):
        super().__init__()
        self.vc: VisualizationContext = visualization_context
        self.data = data
        self.save_figs = False
        self.show_figs = True
        
    def show_input_data(self):
        axs = self.data.df_occupancy.plot(subplots=True)
        axs[-1].axvline(self.data.new_icu_date,color="black",linestyle="--",label="First ICU")

        plt.suptitle("Incidences and ICU Occupancy")
        self._show()

    def plot_icu_data(self):
        fig,ax = self._get_subplots(2,1,figsize=(10,5),sharex=True)

        self.data.df_occupancy["new_icu_smooth"].plot(ax=ax[1],label="new_icu",color="orange")
        self.data.df_occupancy["icu"].plot(ax=ax[0],label="AnzahlFall")
        ax[0].set_title("Tägliche Neuzugänge ICU, geglättet")
        ax[1].set_title("ICU Bettenbelegung")
        plt.tight_layout()
        self._show()
    def plot_mutant_data(self):
        self.data.df_mutant.plot()
        plt.show()
class DeconvolutionPlots (VisualizerBase):

    def __init__(self, all_fit_results: MultiSeriesFitResults, series_data: SeriesData, params:Params, visualization_context:VisualizationContext):
        super().__init__()

        self.vc: VisualizationContext = visualization_context
        self.all_fit_results: MultiSeriesFitResults = all_fit_results
        self.series_data: SeriesData = series_data
        self.params: Params = params


        self.figures_folder = self.vc.figures_folder

        self.save_figs = True
        self.show_figs = True

        
        
        
        
    
        
    def plot_successful_fits(self):
        """Plot number of failed fits and successfull fits"""
        fit_results = self.all_fit_results

        fig = self._figure()

        for i,distro in enumerate(fit_results):
            plt.bar(i,fit_results[distro].n_success,color=self.colors[i],label=distro.capitalize())

        plt.xticks(np.arange(len(fit_results)),fit_results.keys(),rotation=45)
        self._set_title(f"Number of successful fits\n")
        plt.axhline(self.series_data.n_windows,color="red",linestyle="--",label="Total")
        plt.xticks(rotation=45)

        self._show("successful_fits.png", fig)

        
    # Visualization
    def _pairplot(self, col2, col1):
        name = f"{col2}_vs_{col1}"

        fig = self._figure()

        for i, distro in enumerate(self.all_fit_results.distros):
            if distro in ["sentinel","block"]:
                continue
            val1 = self.all_fit_results.summary[col1][distro]
            val2 = self.all_fit_results.summary[col2][distro]
            plt.scatter(val1, val2, s=100, label=distro, color=self.colors[i])
            plt.annotate(distro, (val1, val2), fontsize=9, xytext=(5,5), textcoords='offset points')


        # Labels and formatting
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.grid(True)
        self._set_title(f"Model Performance: {name}")
        self._show(name or f"{col2}_vs_{col1}.png",fig)

    def plot_err_failure_rates(self):
        self._pairplot("Median Loss Train","Failure Rate")
        self._pairplot("Median Loss Test","Failure Rate")
        self._pairplot("Mean Loss Test (no outliers)","Failure Rate")

    def plot_error_comparison(self):
        sorted_summary = self.all_fit_results.summary.sort_values("Median Loss Test")
        sorted_summary = sorted_summary[["Median Loss Test","Failure Rate","Median Loss Train","Upper Quartile Train","Lower Quartile Train"]]

        sorted_summary.plot(subplots=True,figsize=(10,10))

        plt.legend()
        plt.title("Median Loss")
        xticks = list(sorted_summary.index)
        plt.xticks(np.arange(len(xticks)),xticks,rotation=45)
        self._set_title("Error Comparison of Models")
        self._show("error_comparison.png")

    def boxplot_errors(self,errors,title,ylabel,file):
        self._figure()
        plt.boxplot(errors)
        distro_and_n = [f"{distro.capitalize()} n={fr.n_success}" for distro, fr in self.all_fit_results.items()]
        plt.xticks(np.arange(len(distro_and_n))+1,distro_and_n,rotation=45)
        plt.title(title)
        plt.ylabel(ylabel)
        plt.tight_layout()
        self._show(file)

    def stripplot_errors(self, title, file):
        self._figure()
        sns.stripplot(data=self.all_fit_results.train_errors_by_distro, jitter=0.2)
        plt.xticks(np.arange(len(self.all_fit_results)),self.all_fit_results.distros,rotation=45)
        self._set_title(title)
        self._show(file)
        
    def _ax_plot_prediction_error_window(self,ax,fr_series,distro,error_window_alpha=0.1):
        ax.plot(self.series_data.y_full, color="black", alpha=.8,linestyle="--")
        for w,fit_result in zip(fr_series.window_infos,fr_series.fit_results):
            
            if not fit_result.success and error_window_alpha > 0:
                ax.axvspan(w.train_start,w.train_end, color="red", alpha=error_window_alpha)
                continue

            x = np.arange(w.train_los_cutoff,w.train_end)
            y = fit_result.curve[self.params.los_cutoff:self.params.train_width]
            ax.plot(x, y, color=self.colors[0])

            x = np.arange(w.train_end,w.test_end)
            y = fit_result.curve[self.params.train_width:self.params.train_width+self.params.test_width]
            ax.plot(x, y, color=self.colors[1])

        legend_handles = [
            plt.Line2D([0], [0], color='black', linestyle="--", label="Real"),
            plt.Line2D([0], [0], color=self.colors[0], label=f"{distro.capitalize()} Train"),
            plt.Line2D([0], [0], color=self.colors[1], label=f"{distro.capitalize()} Prediction")
        ]
        if error_window_alpha > 0:
            legend_handles += [Patch(color="red", alpha=0.1, label="Failed Training Windows")]
        ax.legend(handles=legend_handles, loc="upper right")

        ax.set_ylim(-100,6000)
        ax.set_xticks(self.vc.xtick_pos[::2])
        ax.set_xticklabels(self.vc.xtick_label[::2])
        ax.set_xlim(*self.vc.xlims)
        ax.grid(zorder=0)


    def _ax_plot_error_error_points(self, ax2, fr_series, distro):
        print ("Warning: _ax_plot_error_error_points has an error and is skipped")
        return
        x = self.series_data.windows   
        ax2.plot(x, fr_series.train_relative_errors,label = "Train Error")
        ax2.plot(x, fr_series.test_relative_errors,label = "Test Error")

        for i, fit_result in enumerate(fr_series.fit_results):
            if not fit_result.success:
                ax2.axvline(x[i],color="red",alpha=.5)
        
        legend_handles = [
            plt.Line2D([0], [0], color=self.colors[0], label=f"{distro.capitalize()} Train"),
            plt.Line2D([0], [0], color=self.colors[1], label=f"{distro.capitalize()} Prediction"),
            plt.Line2D([0], [0], color="red", label="Failed Training Windows", alpha=0.5)
        ]
        ax2.legend(handles=legend_handles, loc="upper right")
        
        ax2.set_ylim(-0.1,.5)
        ax2.set_title("Relative Errors")

        ax2.grid(zorder=0)

    def show_error_windows(self, distro: Optional[Union[str, List[str]]] = None):
        distros = self._get_distro_as_array(distro)
        
        params = self.params

        for distro in distros:
            fr_series = self.all_fit_results[distro]
            _,(ax,ax2) = self._get_subplots(2, 1,sharex=True,figsize=(12,6))
            self._ax_plot_prediction_error_window(ax, fr_series, distro)
            self._ax_plot_error_error_points(ax2, fr_series, distro)

            plt.suptitle(f"{distro.capitalize()} Distribution\n{self.params.run_name}")
            plt.tight_layout()

            self._show(f"prediction_error_{distro}_fit.png")
    def show_all_error_windows_superimposed(self):
        _,(ax,ax2) = self._get_subplots(2, 1,sharex=True,figsize=(12,6))
        for distro in self.all_fit_results.distros:
            fr_series = self.all_fit_results[distro]

            self._ax_plot_prediction_error_window(ax, fr_series, distro,error_window_alpha=.05)
            self._ax_plot_error_error_points(ax2, fr_series, distro)

        for line in ax2.get_children():
            if isinstance(line, plt.Line2D):
                if line.get_color() == 'red':
                    line.remove()
                    
        self._set_title("All Predictions and Error")
        plt.tight_layout()
        self._show(f"prediction_error_all_distros.png")
    
    def _get_distro_as_array(self,distro: Optional[Union[str, List[str]]] = None) -> List[str]:
        if distro is None:
            distros = self.all_fit_results.distros
        elif isinstance(distro, str):
            distros = [distro]
        else:
            distros = distro
        return distros

    def show_all_predictions(self):
        _,ax = self._get_subplots(1, 1,sharex=True,figsize=(15,7.5))
        for distro in self.all_fit_results.distros:
            fr_series = self.all_fit_results[distro]

            self._ax_plot_prediction_error_window(ax, fr_series, distro,error_window_alpha=0)

        self._set_title("All Predictions")
        plt.tight_layout()
        self._show("prediction_all_distros.png")

    def superimpose_kernels(self, distro: Optional[Union[str, List[str]]] = None):
        distros = self._get_distro_as_array(distro)

        for distro in distros:
            self._figure(figsize=(10, 5))
            # plot real kernel
            r, = plt.plot(self.vc.real_los, color='black', label="Real")

            fit_results = self.all_fit_results[distro]
            for fit_result in fit_results.fit_results:
                if not fit_result.success:
                    continue
                l, = plt.plot(fit_result.kernel, alpha=0.3, color=self.colors[0], label="All Estimated")
            plt.legend(handles=[r, l])
            plt.ylim(-0.005, 0.3)
            self._set_title(f"{distro.capitalize()} Kernel")
            plt.tight_layout()
            plt.grid()
            self._show(f"all_kernels_{distro}.png")
        




    def generate_plots_for_run(self,show_plots:Optional[bool]=None,save_figs:Optional[bool]=None):
        if save_figs is not None:
            self.save_figs = save_figs
        if show_plots is not None:
            self.show_figs = show_plots
            
        self.plot_successful_fits()
        self.plot_err_failure_rates()
        self.plot_error_comparison()
        self.boxplot_errors(self.all_fit_results.train_errors_by_distro,"Train Error","Relative Train Error","train_error_boxplot.png")
        self.boxplot_errors(self.all_fit_results.test_errors_by_distro, "Test Error", "Relative Test Error", "test_error_boxplot.png")
        self.stripplot_errors("Train Error", "train_error_stripplot.png")
        self.show_error_windows()
        self.show_all_error_windows_superimposed()
        self.show_all_predictions()
        self.superimpose_kernels()

class DeconvolutionAnimator(DeconvolutionPlots):
    @classmethod
    def from_deconvolution_plots(cls, deconv_plot_visualizer: DeconvolutionPlots):
        """Create an instance of DeconvolutionAnimator from an existing DeconvolutionPlots instance."""
        return cls(
            all_fit_results=deconv_plot_visualizer.all_fit_results,
            series_data=deconv_plot_visualizer.series_data,
            params=deconv_plot_visualizer.params,
            visualization_context=deconv_plot_visualizer.vc
        )

    def __init__(self, all_fit_results, series_data, params, visualization_context):
        super().__init__(all_fit_results, series_data, params, visualization_context)
        self._generate_animation_context()
        self.DEBUG_ANIMATION = False
        self.DEBUG_HIDE_FAILED = False
    
    def DEBUG_MODE(self,
            DEBUG_ANIMATION: bool = True,
            DEBUG_HIDE_FAILED: bool = True,
            SHOW_PLOTS: bool = True,
            SAVE_PLOTS: bool = False,
        ):
        """Set debug configuration for the animator."""
        self.DEBUG_ANIMATION = DEBUG_ANIMATION
        self.DEBUG_HIDE_FAILED = DEBUG_HIDE_FAILED
        self.show_figs = SHOW_PLOTS
        self.save_figs = SAVE_PLOTS
    
        


    def _generate_animation_context(self):    
        alternative_names = {"block":"Constant Discharge","sentinel":"Baseline: Sentinel"}
        replace_short_names =  {"exponential":"exp","gaussian":"gauss","compartmental":"comp"}
        short_distro_names = [distro if distro not in replace_short_names else replace_short_names[distro] for distro in self.all_fit_results]

        distro_colors = {distro: self.vc.graph_colors[i] for i, distro in enumerate(self.all_fit_results)}
        distro_patches = [
            Patch(color=distro_colors[distro], label=alternative_names.get(distro, distro.capitalize()))
            for distro in self.all_fit_results
        ]

        ac = types.SimpleNamespace()
        ac.alternative_names = alternative_names
        ac.distro_colors = distro_colors
        ac.short_distro_names = short_distro_names
        ac.distro_patches = distro_patches        
        self.animation_context = ac



    def _get_subplots(self, SHOW_MUTANTS):
        fig = self._figure(figsize=(17, 10))
        if SHOW_MUTANTS:
            gs = gridspec.GridSpec(3, 4,height_ratios=[5,1,3])
            ax_main = fig.add_subplot(gs[0, :4])
            ax_inc = ax_main.twinx()
            ax_kernel = fig.add_subplot(gs[2,:2])
            ax_err_train = fig.add_subplot(gs[2, 2])
            ax_err_test = fig.add_subplot(gs[2, 3])
            ax_mutant = fig.add_subplot(gs[1, :4])
        else:
            gs = gridspec.GridSpec(2, 4,height_ratios=[2,1])
            ax_main = fig.add_subplot(gs[0, :4])
            ax_inc = ax_main.twinx()
            ax_kernel = fig.add_subplot(gs[1,:2])
            ax_err_train = fig.add_subplot(gs[1, 2])
            ax_err_test = fig.add_subplot(gs[1, 3])
            ax_mutant = None
            
        return ax_main, ax_inc, ax_kernel, ax_err_train, ax_err_test, ax_mutant
    
    def _create_animation_folder(self):
        path = self.vc.animation_folder
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        os.makedirs(path)

    def _plot_ax_main(self,ax_main,ax_inc,window_id):
        w,ac,y_full,x_full = self.series_data.get_window_info(window_id), self.animation_context, self.series_data.y_full, self.series_data.x_full

        
        line_bedload, = ax_main.plot(y_full, color="black",label="ICU Bedload")

        span_los_cutoff = ax_main.axvspan(w.train_start, w.train_los_cutoff, color="magenta", alpha=0.1,label=f"Train Window (Convolution Edge) = {self.params.train_width} days")
        span_train = ax_main.axvspan(w.train_los_cutoff, w.train_end, color="red", alpha=0.2,label=f"Training = {self.params.train_width-self.params.los_cutoff} days")
        span_test = ax_main.axvspan(w.test_start, w.test_end, color="blue", alpha=0.05,label=f"Test Window = {self.params.test_width} days")
        ax_main.axvline(w.train_end,color="black",linestyle="-",linewidth=1)


        for distro,result_series in self.all_fit_results.items():
            result_obj = result_series.fit_results[window_id]
            if self.DEBUG_HIDE_FAILED and not result_obj.success:
                continue
            
            y = result_obj.curve[self.params.los_cutoff:]
            s = np.arange(len(y))+self.params.los_cutoff+w.train_start
            ax_main.plot(s,y, label=f"{distro.capitalize()}", color=ac.distro_colors[distro])


        label = "COVID Incidence (Scaled)"
        if self.params.fit_admissions:
            label = "New ICU Admissions (Scaled)"
        
        

        line_inc, = ax_inc.plot(x_full,linestyle="--",label=label)
        ax_inc.ticklabel_format(axis="y",style="sci",scilimits=(0,0))
        ma = np.nanmax(x_full)
        ax_inc.set_ylim(-ma/7.5,ma*4)


        legend1 = ax_main.legend(handles=ac.distro_patches, loc="upper left", fancybox=True,ncol=2)
        legend2 = ax_main.legend(handles = [line_bedload, line_inc, span_los_cutoff, span_train, span_test],loc="upper right")

        ax_main.add_artist(legend1)
        ax_main.add_artist(legend2)

        ax_main.set_title(f"ICU Occupancy")
        ax_main.set_xticks(self.vc.xtick_pos)
        ax_main.set_xticklabels(self.vc.xtick_label)
        ax_main.set_xlim(*self.vc.xlims)
        ax_main.set_ylim(-200,6000)
        ax_main.set_ylabel("Occupied Beds")

        ax_inc.set_ylabel("(Incidence)")
        if self.params.fit_admissions:
            ax_inc.set_ylabel("New ICU Admissions (scaled)")

    def save_n_show_animation_frame(self, fig: plt.Figure, window_id: int):
        """Save the current figure as an animation frame."""
        if self.DEBUG_ANIMATION:
            plt.show()
        else:
            fig.savefig(self.vc.animation_folder + f"fit_{window_id:04d}.png")
            plt.close(fig)
        plt.clf()


    def _show(self,filename: str=None, fig: Optional[plt.Figure] = None):
        """Save the figure and show it."""
        if fig is None:
            fig = plt.gcf()

        if self.save_figs:
            if not filename.endswith('.png'):
                filename = filename + '.png'
            fig.savefig(self.figures_folder + filename, bbox_inches='tight')

        if self.show_figs:
            plt.show()
        else:
            plt.clf()

    def animate_fit_deconvolution(
            self,
            df_mutant:Optional[pd.DataFrame] = None            
            ):        
        ac = self.animation_context
        y_full = self.series_data.y_full

        SHOW_MUTANTS = df_mutant is not None


        if not self.DEBUG_ANIMATION:
            self._create_animation_folder()
            

        window_counter = 1
        to_enumerate = list(enumerate(self.series_data.window_infos))

        if self.DEBUG_ANIMATION:
            to_enumerate = [to_enumerate[min(2,len(to_enumerate)-1)]]

        for window_id, window_info in to_enumerate:
            print(f"Animation Window {window_counter}/{self.series_data.n_windows}")
            window_counter+=1

            w = window_info
            ax_main, ax_inc, ax_kernel, ax_err_train, ax_err_test, ax_mutant = self._get_subplots(SHOW_MUTANTS)
      
            self._plot_ax_main(ax_main, ax_inc, window_id)
            self._plot_ax_kernel(ax_kernel,window_id)
            self._plot_ax_errors(ax_err_train, ax_err_test,window_id)
            if SHOW_MUTANTS:
                self._plot_ax_mutants(ax_mutant,df_mutant)

            plt.suptitle(f"Deconvolution Training Process\n{self.params.run_name.replace('_',' ')}",fontsize=16)

            plt.tight_layout()
            if self.DEBUG_ANIMATION:
                plt.show()
            else:
                plt.savefig(self.vc.animation_folder + f"fit_{w.train_start:04d}.png")
                plt.close()
            plt.clf()

    def _plot_ax_mutants(self, ax_mutant, df_mutant):
        y_full = self.series_data.y_full
        mutant_lines = []
        for col in df_mutant.columns:
            line, = ax_mutant.plot(df_mutant[col].values)
            mutant_lines.append(line)
            ax_mutant.fill_between(range(len(y_full)), df_mutant[col].values, 0, alpha=0.3)

        ax_mutant.legend(mutant_lines,df_mutant.columns,loc="upper right")
        ax_mutant.set_xticks([])
        ax_mutant.set_xticklabels([])
        ax_mutant.set_xticks(self.vc.xtick_pos)
        tmp_xtick = [label.split("\n")[1:] for label in self.vc.xtick_label]
        tmp_xtick = [label[0] if label else ""  for label in tmp_xtick]
        ax_mutant.set_xticklabels(tmp_xtick)
        ax_mutant.set_xlim(*self.vc.xlims)
        ax_mutant.set_title("Variants of Concern")
        ax_mutant.set_ylabel("Variant Share (%)")

    def _plot_ax_errors(self, ax_err_train, ax_err_test,window_id):
        ac = self.animation_context
        for i, (distro, fit_result) in enumerate(self.all_fit_results.items()):
            c = ac.distro_colors[distro]
            train_err = fit_result.train_relative_errors[window_id]
            test_err = fit_result.test_relative_errors[window_id]

            if self.DEBUG_HIDE_FAILED and not fit_result[window_id].success:
                ax_err_train.bar(i,1e100,color="lightgrey",hatch="/")
                ax_err_test.bar(i, 1e100,color="lightgrey",hatch="/")
                ax_err_train.bar(i,train_err,color="black")
                ax_err_test.bar(i,test_err,color="black")
                continue
            ax_err_train.bar(i,train_err,label="Train",color=c)
            ax_err_test.bar(i,test_err,label="Test",color=c)

        lim = .4
        ax_err_train.set_ylim(0,lim)
        ax_err_train.set_title("Relative Train Error")
        ax_err_train.set_xticks(range(len(self.all_fit_results)))
        ax_err_train.set_xticklabels(ac.short_distro_names,rotation=75)
        ax_err_train.set_ylabel("Relative Error")

        ax_err_test.set_ylim(0,lim)
        ax_err_test.set_title("Relative Test Error")
        ax_err_test.set_xticks(range(len(self.all_fit_results)))
        ax_err_test.set_xticklabels(ac.short_distro_names,rotation=75)
        ax_err_test.set_ylabel("Relative Error")

    def _plot_ax_kernel(self,ax_kernel,window_id):
        ac = self.animation_context
        for distro,result_series in self.all_fit_results.items():
            result_obj = result_series.fit_results[window_id]
            name = ac.alternative_names.get(distro,distro.capitalize())
            if self.DEBUG_HIDE_FAILED and not result_obj.success:
                continue
                
            ax_kernel.plot(result_obj.kernel, label=name, color=ac.distro_colors[distro])

        ax_kernel.plot(self.vc.real_los, color="black",label="Sentinel LoS Charité")
            
        ax_kernel.legend(handles=ac.distro_patches, loc="upper right", fancybox=True, ncol=2, )
        ax_kernel.set_ylim(0,0.1)
        ax_kernel.set_xlim(-2,80)
        ax_kernel.set_ylabel("Discharge Probability")
        ax_kernel.set_xlabel("Days after admission")
        ax_kernel.set_title(f"Estimated LoS Kernels")
