# %%

# %load_ext autoreload
# %autoreload 2
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from los_estimator.core import SeriesData
from util.comparison_data_loader import load_comparison_data

print("Let's Go!")
# %%
less_windows = False
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

        train_error_diff = np.abs(fit_result.train_errors.mean() - comp_fit_result.train_errors.mean())
        test_error_diff = np.abs(fit_result.test_errors.mean() - comp_fit_result.test_errors.mean())

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
        return fit_result.train_errors, comp_fit_result.train_errors


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
debug_config.less_windows = False


# def update(obj, **kwargs):
#     for key, value in kwargs.items():
#         setattr(obj, key, value)
#     return obj


# model_config = update(
#     model_config,
#     kernel_width=120,
#     smooth_data=False,
#     train_width=42 + 60,
#     test_width=21,  # 28 * 4
#     step=7,
#     error_fun="mse",
#     reuse_last_parametrization=True,
#     iterative_kernel_fit=True,
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
fit_results = estimator.all_fit_results

print("done.")

# %%
estimator.visualize_results()
print("visualized.")
# %%
fit_results = estimator.all_fit_results
fit_result = fit_results.results[1]
fr = fit_result.fit_results[0]
fr.test_prediction
from los_estimator.fitting.errors import ErrorFunctions
from pathlib import Path
from datetime import datetime
from plotly.subplots import make_subplots
import numpy as np
import plotly.graph_objects as go

# %%

eval_result = estimator.evaluator.result

evaluation_results_train = eval_result.train
evaluation_results_test = eval_result.test
for i_distro, (distro, fit_result) in enumerate(estimator.all_fit_results.items()):
    print(distro)

    for i_window, (single_fit_result, w) in enumerate(zip(fit_result.fit_results, fit_result.window_infos)):
        x = np.arange(w.training_prediction_start, w.train_end)
        y = single_fit_result.train_prediction[w.kernel_width :]

        y_true = estimator.series_data.y_full[w.training_prediction_start : w.train_end]

        plt.plot(x, y)
        plt.plot(x, y_true, linestyle="dashed", alpha=0.5)

    plt.title(f"TRAIN - {distro}")
    plt.show()

    for i_window, (single_fit_result, w) in enumerate(zip(fit_result.fit_results, fit_result.window_infos)):
        x = np.arange(w.test_start, w.test_end)
        y = single_fit_result.test_prediction[w.kernel_width :]
        y_true = estimator.series_data.y_full[w.test_start : w.test_end]

        plt.plot(x, y)
        plt.plot(x, y_true, linestyle="dashed", alpha=0.5)
    plt.title(f"TEST - {distro}")
    plt.show()
evaluation_results_train.shape, evaluation_results_test.shape

# %%


fig = make_subplots(
    rows=4,
    cols=3,
    specs=[
        [{"colspan": 3}, None, None],
        [{"colspan": 3}, None, None],
        [{}, {}, {}],
        [{}, {}, {}],
    ],
    row_heights=[0.6, 0.2, 0.2, 0.2],
    subplot_titles=(f"Training and Prediction", "Rolling Windows", "A", "B", "C", "D", "E", "F"),
)

fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]), row=1, col=1)

fig.add_trace(go.Scatter(x=[1, 2], y=[1, 2]), row=2, col=1)
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 1, 2]), row=3, col=1)
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 1, 2]), row=3, col=2)
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 1, 2]), row=3, col=3)
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 1, 2]), row=4, col=1)
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 1, 2]), row=4, col=2)
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[2, 1, 2]), row=4, col=3)

fig.show()


################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################


# %%

# Generate Figure 2
data_package = estimator.evaluator.window_data_package

series_data: SeriesData = estimator.series_data
error_window_alpha = 0.05
vc = estimator.visualization_context
distros = list(estimator.all_fit_results.keys())


r_kernel = 3
h1 = 0.38
fig = make_subplots(
    rows=4,
    cols=3,
    specs=[
        [
            {
                "colspan": 3,
            },
            None,
            None,
        ],
        [
            {
                "colspan": 3,
            },
            None,
            None,
        ],
        [{}, {}, {}],
        [{}, {}, {}],
    ],
    row_heights=[0.5, 0.0, 0.2, 0.2],
    subplot_titles=["Training and Prediction", ""] + distros,
    vertical_spacing=0.05,
)


###############################################################################
#################  Real Data  #################################################
###############################################################################

x_full = np.arange(len(data_package.series_data.y_full))
fig.add_trace(
    go.Scatter(
        x=x_full,
        y=data_package.series_data.y_full,
        mode="lines",
        line=dict(color="black"),
        name="Real ICU Occupancy",
        opacity=0.8,
        showlegend=True,
        legendgroup="real",
        legendrank=1,
    )
)
train_legend_shown = False

###############################################################################
#################  Predictions  ###############################################
###############################################################################

first_window = True
windows_to_show = (3, 17, 36)
for i_window in windows_to_show:
    for i_distro, (distro, fit_res) in enumerate(list(estimator.all_fit_results.items())):

        # ensure we don't repeat legend entries for many windows
        d = data_package.data[i_distro][i_window]
        fit_result = estimator.all_fit_results[distro].fit_results[i_window]

        (_, y_pred_train, _, y_pred_test, x_train, x_test, w) = d

        fig.add_trace(
            go.Scatter(
                x=x_train,
                y=y_pred_train,
                mode="lines",
                line=dict(color="orange"),
                name="Training Fits",
                # line=dict(color=estimator.visualization_config.colors[0]),
                legendgroup=f"traces",
                showlegend=not train_legend_shown and first_window,
                legendrank=2,
            ),
            col=1,
            row=1,
        )
        train_legend_shown = True

        fig.add_trace(
            go.Scatter(
                x=x_test,
                y=y_pred_test,
                mode="lines",
                legendgroup=f"traces",
                legendgrouptitle_text="Models",
                name=f"{distro.capitalize()} Models",
                showlegend=first_window,
                legendrank=2,
            )
        )
    first_window = False
i_window = windows_to_show[0]

###############################################################################
#################  Rolling Windows  ###########################################
###############################################################################
for i_window in windows_to_show:
    y = h1 + 0.155  # ensure we don't repeat legend entries for many windows
    d = data_package.data[0][i_window]

    (_, y_pred_train, _, y_pred_test, x_train, x_test, w) = d
    for x in (x_train[0], x_train[-1], x_test[-1]):
        # draw a vertical line that spans (and slightly extends) the plotting area so it appears
        # under axis ticks/labels, and put it below data traces
        fig.add_shape(
            type="line",
            x0=x,
            x1=x,
            xref="x",
            y0=y,
            y1=1,
            yref="paper",
            line=dict(color="gray", dash="dash"),
            opacity=0.8,
            legendrank=3,
        )
    fig.add_shape(
        type="rect",
        x0=x_train[0],
        x1=x_train[-1],
        xref="x",
        y0=y,
        y1=1,
        yref="paper",
        fillcolor="orange",
        opacity=0.2,
        line=dict(width=0),
        layer="below",
    )
    fig.add_shape(
        type="rect",
        x0=x_test[0],
        x1=x_test[-1],
        xref="x",
        y0=y,
        y1=1,
        yref="paper",
        fillcolor="blue",
        opacity=0.2,
        line=dict(width=0),
        layer="below",
    )
    # Add text annotations for train and test regions
    fig.add_annotation(
        text=f"Rolling window {i_window}",
        x=(x_train[0] + x_test[-1]) / 2,
        y=y + 0.055,
        yref="paper",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
    )
    fig.add_annotation(
        text="Train",
        x=(x_train[0] + x_train[-1]) / 2,
        y=y + 0.03,
        yref="paper",
        showarrow=False,
    )
    fig.add_annotation(
        text="Pred",
        x=(x_test[0] + x_test[-1]) / 2,
        y=y + 0.03,
        yref="paper",
        showarrow=False,
    )
###############################################################################
#################  LoS Distros  ###############################################
###############################################################################

from los_estimator.fitting.distributions import Distributions

i_window = 3
for i_distro, (distro, fit_res) in enumerate(list(estimator.all_fit_results.items())):
    d = data_package.data[i_distro][i_window]
    fit_result = estimator.all_fit_results[distro].fit_results[i_window]
    kernel = fit_result.kernel
    x_kernel = np.arange(len(kernel))
    fig.add_trace(
        go.Scatter(
            x=x_kernel,
            y=kernel,
            mode="lines",
            showlegend=False,
        ),
        row=r_kernel + i_distro // 3,
        col=1 + i_distro % 3,
    )
    param_str = ""
    if distro != "compartmental":
        param_str = Distributions.to_string(distro, fit_result.model_config).replace(",", "<br>")
        fig.add_annotation(
            text=param_str,
            x=60,
            y=0.051,
            xanchor="right",
            yanchor="top",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1,
            borderpad=4,
            font=dict(size=9),
            row=r_kernel + i_distro // 3,
            col=1 + i_distro % 3,
            align="right",  # Align text to the right
        )

# general_param_str = "<br>".join(
#     [
#         f"<b>{distro.capitalize()}</b>: {Distributions.to_string(distro, fit_res[i_window].model_config)}"
#         for distro, fit_res in list(estimator.all_fit_results.items())
#         if distro != "compartmental"
#     ]
# )

# fig.add_annotation(
#     text=general_param_str,
#     xref="paper",
#     yref="paper",
#     x=0.99,
#     y=0.25,
#     xanchor="right",
#     yanchor="top",
#     showarrow=False,
#     bgcolor="rgba(255,255,255,0.9)",
#     bordercolor="lightgray",
#     borderwidth=1,
#     borderpad=10,
#     font=dict(size=10),
# )


fig.update_yaxes(
    col=1, row=r_kernel, title_text="Discharge Probability", gridwidth=1, gridcolor="lightgray", range=(0, 0.051)
)
###############################################################################
#################  General stuff  #############################################
###############################################################################

# place legend at top-right of the first (top) subplot
fig.update_layout(
    legend=dict(
        x=0.99,
        y=0.99,
        xanchor="right",
        yanchor="top",
        # bgcolor="rgba(255,255,255,0.7)",
        bordercolor="lightgray",
        borderwidth=1,
    )
)


xaxis = dict(
    range=(50, 399),
    tickmode="array",
    tickvals=vc.xtick_pos[1:],
    ticktext=[element.replace("\n", "<br>") for element in vc.xtick_label[1:]],
    gridcolor="lightgray",
)

fig.update_layout(
    height=900,
    width=900,
    xaxis=xaxis,
    yaxis=dict(range=[-100, 5200], title="ICU occupancy"),
    title="Rolling Training and Prediction of ICU Occupancy with Different LoS Distributions",
    template="plotly_white",
)


# add gridlines (Plotly shows grid by default with template; ensure visible)
# fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
fig.update_xaxes(col=1, row=1, **xaxis)


fig.show()


# %%
# for i in range(3):
#     col = 1
#     row = i + 1
#     legend_name = f"legend{i+2}"  # legend1 is the theme's default. start at legend2 to avoid.
#     x = ((col - 1) * (subplot_width + horizontal_spacing)) + (subplot_width / 2)
#     y = 1 - ((row - 1) * (subplot_height + vertical_spacing)) + legend_horizontal_spacing
#
#     fig.update_traces(col=1, row=i + 1, legend_name=f"legend_{i}")
#     fig.update_layout(
#         {
#             legend_name: dict(
#                 x=x,
#                 y=y,
#                 xanchor="center",
#                 yanchor="bottom",
#                 bgcolor="rgba(0,0,0,0)",
#             )
#         }
#     )
#
# fig.update_yaxes(col=1,row=2,title_text="Window Index",gridwidth=1, gridcolor="lightgray")
#
# fig.show()
#
# output_dir = Path(estimator.output_config.figures)
# output_dir.mkdir(parents=True, exist_ok=True)
# filename = output_dir / f"training_prediction_{distro}.html"
# fig.write_html(str(filename), include_plotlyjs="cdn", full_html=True)


################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################
################################################################################################################################################################################################


# %%
import plotly.express as px

# %%
vc = estimator.visualization_context

metric_name = "mse"
i_metric = 0
fig = go.Figure()
for i_distro, distro in eval_result.iter_distros(ret_arr=False):
    import plotly.graph_objects as go

    fit_res = estimator.all_fit_results[distro]
    time_points = [w.window for w in fit_res.window_infos]

    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=evaluation_results_train[i_distro, :, i_metric],
            mode="lines+markers",
            name=f"{distro} (train)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time_points,
            y=evaluation_results_test[i_distro, :, i_metric],
            mode="lines+markers",
            name=f"{distro} (test)",
            line=dict(dash="dash"),
        )
    )

fig.update_layout(
    height=480,
    width=900,
    yaxis=dict(title="MSE"),
    xaxis=dict(
        range=list(vc.xlims),
        tickmode="array",
        tickvals=vc.xtick_pos[1::2],
        ticktext=[element.replace("\n", "<br>") for element in vc.xtick_label[1::2]],
    ),
    title=f"MSE over time",
    template="plotly_white",
)
fig.show()
filename = Path(estimator.output_config.figures) / f"mse_plot.html"

fig.write_html(str(filename), include_plotlyjs="cdn", full_html=True)
print(f"Saved figure to {filename}")
# %%
skip_graph = True
figure_explanation = (
    "For each Distribution, for each metric: plot all fit results, predictions and train metric and test metric"
)
if skip_graph:
    print("Skipping Figure:", figure_explanation)
else:
    print("Generating Figure:", figure_explanation)
    # Figure Explanation: For each Distribution, for each metric: plot all fit results, predictions and train metric and test metric
    # Print estimations
    metrics = eval_result.metric_names
    for i_distro, distro in enumerate(estimator.all_fit_results.keys()):
        if i_distro > 1:
            continue
        fit_res = estimator.all_fit_results[distro]
        time_points = [w.window for w in fit_res.window_infos]

        for i_metric, metric in enumerate(metrics):
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                row_heights=[0.6, 0.4],
                vertical_spacing=0.08,
                subplot_titles=(f"{distro} - Test Predictions", f"{metric}"),
            )

            # Top: test predictions (one trace per window) and true series
            for i_window, (single_fit_result, w) in enumerate(zip(fit_res.fit_results, fit_res.window_infos)):
                x = np.arange(w.test_start, w.test_end)
                y = single_fit_result.test_prediction[w.kernel_width :]
                y_true = estimator.series_data.y_full[w.test_start : w.test_end]

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        name=f"pred w{w.window}",
                        legendgroup=f"w{i_window}",
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y_true,
                        mode="lines",
                        name=f"true w{w.window}",
                        line=dict(dash="dash"),
                        legendgroup=f"w{i_window}",
                    ),
                    row=1,
                    col=1,
                )

            # Bottom: metric over time (train vs test)
            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=evaluation_results_train[i_distro, :, i_metric],
                    mode="lines+markers",
                    name="train",
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=evaluation_results_test[i_distro, :, i_metric],
                    mode="lines+markers",
                    name="test",
                    line=dict(dash="dash"),
                ),
                row=2,
                col=1,
            )

            if len(time_points) > 0:
                fig.update_xaxes(range=[time_points[0], time_points[-1]])

            fig.update_layout(height=700, width=900, showlegend=True)
            fig.show()
# %%
data_package = estimator.evaluator.window_data_package
data_package
distro, i_distro = "gaussian", 0
(
    y_true_train,
    y_pred_train,
    y_true_test,
    y_pred_test,
    x_train,
    x_test,
    w,
) = data_package.data[
    i_distro
][0]
# %%

series_data: SeriesData = estimator.series_data
error_window_alpha = 0.05
vc = estimator.visualization_context

for i_distro, (distro, fit_res) in enumerate(list(estimator.all_fit_results.items())):

    fig = go.Figure()

    # plot real series
    x_full = np.arange(len(data_package.series_data.y_full))
    fig.add_trace(
        go.Scatter(
            x=x_full,
            y=data_package.series_data.y_full,
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="Real ICU Occupancy",
            opacity=0.8,
            showlegend=True,
        )
    )

    # ensure we don't repeat legend entries for many windows
    train_legend_shown = False
    pred_legend_shown = False
    window_legend_shown = False

    for d, fit_result in zip(data_package.data[i_distro], estimator.all_fit_results[distro].fit_results):
        (
            y_true_train,
            y_pred_train,
            y_true_test,
            y_pred_test,
            x_train,
            x_test,
            w,
        ) = d
        if not fit_result.success and error_window_alpha > 0:
            fig.add_shape(
                type="rect",
                xref="x",
                yref="paper",
                x0=w.train_start,
                y0=0,
                x1=w.train_end,
                y1=1,
                fillcolor="red",
                opacity=error_window_alpha,
                line=dict(width=0),
                layer="below",
                name="Error windows",
                legendgroup="error_windows",
                showlegend=not window_legend_shown,
            )
            window_legend_shown = True

        fig.add_trace(
            go.Scatter(
                x=x_train,
                y=y_pred_train,
                mode="lines",
                line=dict(color=estimator.visualization_config.colors[0]),
                name=f"{distro.capitalize()} Training Fits",
                legendgroup=f"{distro}_train",
                showlegend=not train_legend_shown,
            )
        )
        train_legend_shown = True

        fig.add_trace(
            go.Scatter(
                x=x_test,
                y=y_pred_test,
                mode="lines",
                line=dict(color=estimator.visualization_config.colors[1], dash="solid"),
                name=f"{distro.capitalize()} Predictions",
                legendgroup=f"{distro}_pred",
                showlegend=not pred_legend_shown,
            )
        )

        pred_legend_shown = True
        break
    # layout / legend / axes
    fig.update_layout(
        height=480,
        width=900,
        yaxis=dict(range=[-100, 6000], title="ICU occupancy"),
        xaxis=dict(
            range=list(vc.xlims),
            tickmode="array",
            tickvals=vc.xtick_pos[1::2],
            ticktext=[element.replace("\n", "<br>") for element in vc.xtick_label[1::2]],
        ),
        title=f"Training and Prediction {distro.capitalize()} distribution",
        template="plotly_white",
    )

    # add gridlines (Plotly shows grid by default with template; ensure visible)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="lightgray")

    fig.show()
    # save figure to output directory
    output_dir = Path(estimator.output_config.figures)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"training_prediction_{distro}.html"
    fig.write_html(str(filename), include_plotlyjs="cdn", full_html=True)

# %%
sorted_summary = estimator.all_fit_results.summary.sort_values("Median Loss Test")
sorted_summary = sorted_summary[
    [
        "Median Loss Test",
        "Failure Rate",
        "Median Loss Train",
        "Upper Quartile Train",
        "Lower Quartile Train",
    ]
]
import pandas as pd


def get_dfs(self) -> pd.DataFrame:
    """Return a DataFrame with train/test mean and median per distribution and metric."""
    if self.train is None or self.test is None:
        raise RuntimeError("Metrics not available - call calculate_metrics() first")

    # Use precomputed summaries if present, otherwise compute from arrays
    train_mean = getattr(self, "train_mean", None)
    if train_mean is None:
        train_mean = self.train.mean(axis=1)
    test_mean = getattr(self, "test_mean", None)
    if test_mean is None:
        test_mean = self.test.mean(axis=1)

    train_median = getattr(self, "train_median", None)
    if train_median is None:
        train_median = np.median(self.train, axis=1)
    test_median = getattr(self, "test_median", None)
    if test_median is None:
        test_median = np.median(self.test, axis=1)

    rows = []
    for i_distro, distro in enumerate(self.distros):
        for i_metric, metric in enumerate(self.metric_names):
            rows.append(
                {
                    "distribution": distro,
                    "metric": metric,
                    "train_mean": float(train_mean[i_distro, i_metric]),
                    "train_median": float(train_median[i_distro, i_metric]),
                    "test_mean": float(test_mean[i_distro, i_metric]),
                    "test_median": float(test_median[i_distro, i_metric]),
                }
            )

    return pd.DataFrame(rows)


res = estimator.evaluator.result
df_metrics = get_dfs(res)
df_metrics
# %%

metric = "mse"
df = df_metrics[df_metrics["metric"] == metric]
df["failure_rate"] = estimator.all_fit_results.summary["Failure Rate"].values


def do_scatter(x, y):
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color="failure_rate",
        color_continuous_scale="RdYlGn_r",
        size_max=18,
        hover_data=["distribution", "failure_rate"],
        labels={x: x, y: y},
        title="Median Loss: Train vs Test by Distribution",
    )

    # show distribution names on points
    fig.update_traces(textposition="top center", marker=dict(line=dict(width=0.5, color="DarkSlateGrey")))

    # add y=x reference
    vals = df[[x, y]].values
    minv = float(vals.min()) if vals.size else 0.0
    maxv = float(vals.max()) if vals.size else 1.0
    fig.add_shape(
        type="line",
        x0=minv,
        y0=minv,
        x1=maxv,
        y1=maxv,
        line=dict(color="gray", dash="dash"),
    )

    fig.update_layout(height=560, width=800, template="plotly_white", coloraxis_colorbar=dict(title="Failure Rate"))
    # fig.show()


do_scatter("train_mean", "test_mean")
do_scatter("train_median", "test_median")


# # ensure output dir exists and save
# output_dir = Path(estimator.output_config.figures)
# output_dir.mkdir(parents=True, exist_ok=True)
# filename = output_dir / "median_loss_train_vs_test.html"
# fig.write_html(str(filename), include_plotlyjs="cdn", full_html=True)
# print(f"Saved median loss scatter to {filename}")

# # fig.show()

# %%
# Plot all fitted kernels superimposed for each distribution using Plotly

output_dir = Path(estimator.output_config.figures)
output_dir.mkdir(parents=True, exist_ok=True)

for distro, fit_res in list(estimator.all_fit_results.items()):
    # Try to obtain kernels array (windows x kernel_width)
    kernels = getattr(fit_res, "all_kernels", None)
    if kernels is None:
        # fallback: collect from individual fit results if available
        collected = []
        for sf in getattr(fit_res, "fit_results", []):
            k = getattr(sf, "kernel", None)
            if k is None:
                k = getattr(sf, "fitted_kernel", None)
            if k is None:
                k = getattr(sf, "est_kernel", None)
            if k is not None:
                collected.append(np.asarray(k))
        if collected:
            kernels = np.vstack(collected)
        else:
            print(f"⚠️ No kernels found for distribution: {distro}")
            continue

    kernels = np.asarray(kernels)
    if kernels.ndim == 1:
        kernels = kernels[None, :]

    n_windows, kernel_width = kernels.shape
    x = np.arange(kernel_width)

    fig = go.Figure()
    # one trace per window
    win_infos = getattr(fit_res, "window_infos", None)
    for i in range(n_windows):
        label = f"w{i}"
        if win_infos is not None and i < len(win_infos):
            label = f"w{win_infos[i].window}"
        fig.add_trace(
            go.Scatter(
                x=x,
                y=kernels[i],
                mode="lines",
                name="Kernel",
                legendgroup=f"Kernels",
                legendgrouptitle_text="Kernels",
                showlegend=(i == 0),
                line=dict(color="lightblue", width=1),
                opacity=0.1,
                hovertemplate="day=%{x}<br>discharge=%{y:.4f}<extra></extra>",
            )
        )

    # plot mean kernel
    mean_kernel = kernels.mean(axis=0)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=mean_kernel,
            mode="lines",
            name="mean",
            line=dict(color="blue", width=2),
            hovertemplate="lag=%{x}<br>mean=%{y:.4f}<extra></extra>",
        )
    )
    real_los = vc.real_los
    if real_los is not None:
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(real_los)),
                y=real_los,
                mode="lines",
                name="Sample LOS",
                line=dict(color="black", width=2),
                hovertemplate="lag=%{x}<br>Sample LOS=%{y:.4f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"{distro} — fitted kernels (n={n_windows})",
        xaxis_title="lag",
        yaxis_title="kernel value",
        legend_title="window",
        template="plotly_white",
        height=480,
        width=900,
    )

    # fig.show()
    filename = output_dir / f"kernels_{distro}.html"

    # uniform color + opacity for all traces
    uniform_color = "royalblue"
    uniform_opacity = 0.6

    # apply to all traces (windows + mean)
    fig.update_traces(line_color=uniform_color, opacity=uniform_opacity)
    # make the mean kernel (last trace) a bit more prominent
    if len(fig.data) > 0:
        fig.data[-1].update(line=dict(color=uniform_color, width=3), opacity=1.0)

    fig.write_html(str(filename), include_plotlyjs="cdn", full_html=True)
    fig.show()
    print(f"Saved kernels plot for {distro} to {filename}")


# %%
# Show single example kernel
distro_id = 1
for distro, fit_res in list(estimator.all_fit_results.items())[distro_id : distro_id + 1]:
    kernels = getattr(fit_res, "all_kernels", None)
    kernels = np.asarray(kernels)

    i = 0
    kernel = kernels[i]
    parameters = fit_res[i].model_config
    if distro == "gaussian":
        param_str = "stretch={:.2f}, μ={:.2f}, σ={:.2f}".format(*parameters)
    if distro == "exponential":
        param_str = "stretch={:.2f}, 1/λ={:.2f}".format(*parameters)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=kernel,
            name="Discretized LoS",
            marker=dict(color="royalblue", opacity=0.8),
            hovertemplate="days=%{x}<br>mean=%{y:.4f}<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            y=kernel,
            mode="lines",
            name="Continuous Distribution Function",
            line=dict(color="blue", width=2),
            hovertemplate="day=%{x}<br>mean=%{y:.4f}<extra></extra>",
            line_color="black",
        )
    )

    fig.update_layout(
        title=f"{distro} - fitted kernel example, {param_str}",
        xaxis_title="days",
        yaxis_title="discharge probability",
        template="plotly_white",
        height=480,
        width=900,
    )

    fig.show()
    filename = output_dir / f"example_kernel_exponential.html"
# %%
# Show 3x3 grid of examplary kernels

output_dir = Path(estimator.output_config.figures)
output_dir.mkdir(parents=True, exist_ok=True)
fig = make_subplots(
    rows=2, cols=3, row_heights=[0.5, 0.5], vertical_spacing=0.2, subplot_titles=list(estimator.all_fit_results)
)


for i, (distro, fit_res) in enumerate(list(estimator.all_fit_results.items())):
    kernels = np.asarray(fit_res.all_kernels)
    row, col = (i // 3) + 1, (i % 3) + 1
    n_windows, kernel_width = kernels.shape
    x = np.arange(kernel_width)

    # one trace per window
    win_infos = getattr(fit_res, "window_infos", None)
    kernel_id = 0

    fig.add_trace(
        go.Scatter(
            x=x,
            y=kernels[kernel_id],
            mode="lines",
            showlegend=False,
            line=dict(color="black", width=1),
            hovertemplate="day=%{x}<br>discharge=%{y:.4f}<extra></extra>",
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Bar(
            y=kernels[kernel_id],
            marker=dict(color="royalblue", opacity=0.8),
            showlegend=False,
            hovertemplate="days=%{x}<br>mean=%{y:.4f}<extra></extra>",
        ),
        row=row,
        col=col,
    )
fig.update_layout(
    title=f"Distributions — fitted kernel examples t=0",
    legend_title="window",
    template="plotly_white",
    height=500,
    width=500,
)


# fig.show()
filename = output_dir / f"kernels_{distro}.html"

# fig.update_traces(line_color="black", opacity=1)
fig.update_xaxes(matches="x")
fig.update_yaxes(matches="y")

# fig.write_html(str(filename), include_plotlyjs="cdn", full_html=True)
fig.show()
print(f"Saved kernels plot for {distro} to {filename}")


# %%
vc = estimator.visualization_context
time_points = [w.window for w in fit_result.window_infos]
for i_distro, distro in eval_result.iter_distros(ret_arr=False):
    fig, axs = plt.subplots(3, len(metrics) // 2, figsize=(4 * len(metrics), 12), sharex=True)
    axs = axs.flatten()
    for i_metric, metric_name in eval_result.iter_metrics(ret_arr=False):
        ax = axs[i_metric]
        ax.plot(time_points, evaluation_results_train[i_distro, :, i_metric], label="train")
        ax.plot(time_points, evaluation_results_test[i_distro, :, i_metric], label="test")
        ax.set_title(f"{distro} - {metric_name}")
        ax.legend()
    ax.set_xticks(vc.xtick_pos[::2])
    ax.set_xticklabels(vc.xtick_label[::2])
    ax.set_xlim(time_points[0], time_points[-1])

plt.show()


# %%
time_points = [w.window for w in fit_result.window_infos]
for i_distro, distro in enumerate(estimator.all_fit_results.keys()):
    for i_metric, metric in enumerate(metrics):
        fig, axs = plt.subplots(2, 1, figsize=(6, 4), sharex=True)

        ax = axs[0]
        ax.set_title(f"{distro} - Test Predictions")

        for i_window, (single_fit_result, w) in enumerate(zip(fit_result.fit_results, fit_result.window_infos)):
            x = np.arange(w.test_start, w.test_end)
            y = single_fit_result.test_prediction[w.kernel_width :]
            y_true = estimator.series_data.y_full[w.test_start : w.test_end]
            ax.plot(x, y)
            ax.plot(x, y_true, linestyle="dashed", alpha=0.5)

        ax = axs[1]
        ax.plot(time_points, evaluation_results_train[i_distro, :, i_metric], label="train")
        ax.plot(time_points, evaluation_results_test[i_distro, :, i_metric], label="test", linestyle="dashed")
        ax.set_title(f"{metric}")
        ax.legend()

        plt.tight_layout()
        plt.show()

# %%
