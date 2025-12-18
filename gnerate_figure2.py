# %%
# First run run_analysis


# Generate Figure 2
data_package = estimator.evaluator.window_data_package

series_data: SeriesData = estimator.series_data
error_window_alpha = 0.05
vc = estimator.visualization_context
distros = list(estimator.all_fit_results.keys())


r_kernel = 4
h1 = 0.43
wide_plot = [
    {
        "colspan": 3,
    },
    None,
    None,
]
fig = make_subplots(
    rows=5,
    cols=3,
    specs=[
        wide_plot,
        wide_plot,
        wide_plot,
        [{}, {}, {}],
        [{}, {}, {}],
    ],
    row_heights=[0.5, 0, 0, 0.2, 0.2],
    subplot_titles=["Training and Prediction", "", "Estimated Kernels at position n=3"] + distros,
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
    y_below = y - 0.012
    n = "n"
    li = [
        (f"t{n}-Δtrain", x_train[0]),
        (f"t{n}", x_test[0]),
        (f"t{n}+Δpred", x_test[-1]),
    ]
    for text, x in li:
        fig.add_annotation(
            text=text,
            x=x,
            y=y_below,
            yref="paper",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
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


output_dir = Path(estimator.output_config.figures)
output_dir.mkdir(parents=True, exist_ok=True)
filename = output_dir / f"figure_2.html"
fig.write_html(str(filename), include_plotlyjs="cdn", full_html=True)
