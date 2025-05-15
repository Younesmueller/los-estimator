#%%
# reload imports
%load_ext autoreload
%autoreload 2
import os
import sys
import numpy as np
import pandas as pd
import types
import seaborn as sns

import time

import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sys.path.append("../02_fit_los_distributions/")
from dataprep import load_los, load_incidences, load_icu_occupancy, load_age_groups
from los_fitter import fit_kernel_distro_to_data, distributions, calc_its_convolution
plt.rcParams['savefig.facecolor']='white'
os.chdir(os.path.dirname(os.path.abspath(__file__)))



#%%
print("Let's Go!")
#%%
print_figs = True
def show_plt(*args,**kwargs):
    # if in vscode interactive
    if print_figs:
        plt.show()
    else:
        plt.clf()

#%%
los_file = "../01_create_los_profiles/berlin/output_los/los_berlin_all.csv"

kernel_width = 120
los_cutoff = 60 # Ca. 90% of all patients are discharged after 41 days

start_day = "2020-01-01"
end_day = "2023-09-25"

def date_to_day(date):
    return (date - pd.Timestamp(start_day)).days
def day_to_date(day):
    return pd.Timestamp(start_day) + pd.Timedelta(days=day)

#%%
# take matplotlib standart color wheel
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# add extra color palette
colors += ["#FFA07A","#20B2AA","#FF6347","#808000","#FF00FF","#FFD700","#00FF00","#00FFFF","#0000FF","#8A2BE2"]

#%%
real_los, _ = load_los(file=los_file)
df_inc, raw = load_incidences(start_day, end_day)
# wenn die ersten beiden Yeilen der 1.1.2020 sind entferne eine
if df_inc.index[0] == pd.Timestamp("2020-01-01") and df_inc.index[1] == pd.Timestamp("2020-01-01"):
    df_inc = df_inc.iloc[1:]

df_icu = load_icu_occupancy(start_day, end_day)
df = df_inc.join(df_icu,how="inner")
df["new_icu_smooth"] = df["new_icu"].rolling(7).mean()

sentinel_start_date =pd.Timestamp("2020-10-01")
sentinel_end_date = pd.Timestamp("2021-06-21")
sentinel_start_day = date_to_day(sentinel_start_date)
sentinel_end_day = date_to_day(sentinel_end_date)

new_icu_date = df.index[df["new_icu"]>0][0]
new_icu_day = date_to_day(new_icu_date)

axs = df.plot(subplots=True)
for ax in axs:
    ax.axvspan(sentinel_start_date,sentinel_end_date, color="green", alpha=0.1,label="Sentinel")
axs[-1].axvline(new_icu_date,color="black",linestyle="--",label="First ICU")

plt.suptitle("Incidences and ICU Occupancy")
show_plt()
#%%
# load hospitalization profile
df_hosp_ag = pd.read_csv("../data/hosp_ag.csv",parse_dates=["Datum"])
df_hosp_ag.set_index("Datum", inplace=True)
assignment= {
    "0_60":[
        '00-04',
        '05-14',
        '15-34',
        '35-59',],
    "60_79":['60-79',],
    "80_100":['80+'],
}
for k,v in assignment.items():
    df_hosp_ag[k] = df_hosp_ag[v].sum(axis=1)
    df_hosp_ag = df_hosp_ag.drop(columns=v)

date_range = pd.date_range(start="2020-01-01", end=df_hosp_ag.index.min(),inclusive='left')
new_data = pd.DataFrame(0, index=date_range, columns=df_hosp_ag.columns)
df_hosp_ag = pd.concat([new_data, df_hosp_ag])
# cut off at start end end date
df_hosp_ag = df_hosp_ag[df_hosp_ag.index >= start_day]
df_hosp_ag = df_hosp_ag[df_hosp_ag.index <= end_day]

df_hosp_ag
df_hosp_relative = df_hosp_ag.copy()
sum_df = df_hosp_relative.sum(axis=1)
df_hosp_relative[sum_df==0] += 1/(len(df_hosp_relative.columns))
sum_df = df_hosp_relative.sum(axis=1)
for col in df_hosp_relative.columns:
    df_hosp_relative[col] /= sum_df
df_hosp_relative.plot()
plt.show()
df_hosp_relative

#%%
# Calculate admissions by age group
df_icu_ag = df_hosp_relative.copy()
for col in df_icu_ag.columns:
    df_icu_ag[col] *= df["new_icu"]
plt.plot(df_icu_ag)
#%%
from dataprep import load_age_groups
df_age = load_age_groups(start_day, end_day)
df_age.plot()
plt.show()
assignment= {
    "0_60":[
        'altersgruppe_0_bis_17',
        'altersgruppe_18_bis_29',
        'altersgruppe_30_bis_39',
        'altersgruppe_40_bis_49',
        'altersgruppe_50_bis_59', ],
    "60_70":['altersgruppe_60_bis_69',],
    "70_100":['altersgruppe_70_bis_79','altersgruppe_80_plus',],
}
for k,v in assignment.items():
    df_age[k] = df_age[v].sum(axis=1)
    df_age = df_age.drop(columns=v)
df_age= df_age.drop(columns=["altersgruppe_unbekannt"])
df_age.plot()
plt.show()
n_ag = len(df_age.columns)
#%%
xtick_pos = []
xtick_label = []
for i in range(0,len(df)):
    # take only first of month
    if df.index[i].day == 1:
        xtick_pos.append(i)
        # append month name
        label = df.index[i].strftime("%b")
        if i == 0 or df.index[i].month == 1:
            label += f"\n{df.index[i].year}"
        xtick_label.append(label)


#%%
fig,axs =  plt.subplots(3,1,figsize=(10,5),sharex=True)
(ax1,ax2,ax3) = axs
ax1.plot(df["icu"].values,                 label="Real ICU Bedload")
ax2.plot(df["AnzahlFall"].values,          label="Real Incidence")
ax3.plot(df["icu"].values/df["AnzahlFall"].values,label="Real ICU Bedload / Incidence")
ax1.set_title("ICU Bedload")
ax2.set_title("Incidence")
ax3.set_title("ICU Bedload / Incidence")
for ax in axs:
    ax.grid()
    ax.legend()
ax1.set_xticks(xtick_pos)
ax1.set_xticklabels(xtick_label)
ax1.set_xlim(140,270)
ax1.set_ylim(-100,500)
ax2.set_ylim(-100,2000)

plt.show()


#%%
# Load init parameters
file = "../02_fit_los_distributions/output_los/los_berlin_all/fit_results.csv"
df_init = pd.read_csv(file,index_col=0)
df_init = df_init.set_index("distro")
# interpret params as array float of format [f1 f2 f3 ...]
df_init["params"] = df_init["params"].apply(lambda x: [float(i) for i in x[1:-1].split()])
df_init



#%%



params = types.SimpleNamespace()
params.use_manual_transition_rate = False
params.use_gaussian_spread = False
params.smooth_data = False
params.train_width = 7 + los_cutoff
params.test_width = 21 #28 *4
params.step = 7
params.fit_admissions = True
params.fit_admission_age_groups = True
params.use_ag_input_data = True

params
#%%
timestamp = time.strftime("%y%m%d_%H%M")
run_name = f"{timestamp}_dev"

# for grid_search_id in range(0,12):

    # run_name = f"{timestamp}_grid_search"

    # params = types.SimpleNamespace()
    # # generate params from run_id
    # params.train_width = [7 + los_cutoff, 14 + los_cutoff, 21 + los_cutoff]
    # params.test_width = [7]
    # params.step = [21]
    # params.use_manual_transition_rate = [False, True]
    # params.use_gaussian_spread = [False, True]
    # params.fit_admissions = [True]

    # param_space = list(itertools.product(
    #     *params.__dict__.values()
    # ))
    # # set params dict values
    # for i,key in enumerate(params.__dict__.keys()):
    #     params.__dict__[key] = param_space[grid_search_id][i]


run_name+=f"_step{params.step}_train{params.train_width}_test{params.test_width}"
if params.use_manual_transition_rate:
    run_name += "_manual_transition"
if params.use_gaussian_spread:
    run_name += "_gaussian_spread"
if params.fit_admissions:
    run_name += "_fit_admissions"
else:
    run_name += "_fit_incidence"
if params.smooth_data:
    run_name += "_smoothed"
else:
    run_name += "_unsmoothed"

if params.use_ag_input_data:
    run_name += "_use_ag_input_data"

if params.fit_admission_age_groups:
    run_name += "_fit_age_groups"
print(run_name)
results_folder = f"./results/{run_name}/"
figures_folder = results_folder + "/figures/"
animation_folder = results_folder + "/animation/"

if os.path.exists(results_folder):
    import shutil
    shutil.rmtree(results_folder)
os.makedirs(results_folder)
os.makedirs(figures_folder)
os.makedirs(animation_folder)

start = 0
if params.fit_admissions:
    start = new_icu_day
windows = np.arange(start,len(df)-params.step,params.step)
windows = np.arange(start,800,params.step)
# remove windows, where y_test is too short
windows = np.array([window for window in windows if window+params.train_width+params.test_width < len(df)])+params.train_width

class WindowInfo:
    def __init__(self,window):
        self.window = window
        self.train_end = self.window
        self.train_start = self.window - params.train_width
        self.train_los_cutoff = self.train_start + los_cutoff
        self.test_start = self.train_end
        self.test_end = self.test_start + params.test_width


# Specify transition rate manually
manual_points = np.array(
    [[0,               5.26125540e-02],
    [ 4.79999995e+01,  5.29817366e-02],
    [ 6.89999731e+01,  5.25110807e-02],
    [ 1.10999986e+02,  5.42751008e-02],
    [ 1.73999994e+02,  3.54917868e-02],
    [ 2.37000003e+02,  1.42336509e-02],
    [ 4.99999999e+02,  1.48902533e-02],
    [ 5.94000000e+02,  1.08504095e-02],
    [ 7.83000001e+02,  1.49297704e-03],
    [ 1.11900000e+03,  8.93606945e-04],
    [ 1.32900000e+03,  1.76263698e-02]]
)
xs = np.arange(len(df))
manual_transition_rates = np.interp(xs,manual_points[:,0],manual_points[:,1])
plt.plot(manual_transition_rates,label="Manual Transition Rates")
plt.title("Manual Transition Rates")
plt.xticks(xtick_pos[::4],xtick_label[::4])
show_plt()




#%%
# Fitting the LoS curves, as well as the delay and probability
from scipy.optimize import minimize
from los_fitter import mse, generate_kernel, distributions
from los_fitter import boundaries as boundaries2
def calc_age_convolution(x, distro, kernel_width, ag_distro, distro_params,los_cutoff):
    n = len(ag_distro)
    ag_distro[-1] = 1- np.sum(ag_distro[:-1])
    observed = np.zeros((n,len(x)))
    for i in range(n):
        kernel = generate_kernel(distro, distro_params[i], kernel_width)
        observed[i] = calc_its_convolution(
            inc=x*ag_distro[i],
            los_distro1=kernel,
            transition_rate=1,
            delay=0,
            los_cutoff=los_cutoff)

    return observed

def objective_age_group_fit(distro,kernel_width,los_cutoff,n):
    def objective_function(params, x, y):
        ag_distro = params[:n]
        distro_params = params[n:]
        distro_params = np.reshape(distro_params,(n,-1))
        observed = calc_age_convolution(x, distro, kernel_width, ag_distro, distro_params,los_cutoff)

        return mse(y.T[:,los_cutoff:], observed[:,los_cutoff:])


    return objective_function

def fit_age_group_distros(
    distro,
    x_train,
    y_train,
    x_test,
    y_test,
    kernel_width,
    los_cutoff,
    ag_distro_init_params,
    distro_init_params,
    distro_boundaries=None,
    method="L-BFGS-B",
):

    if distro_boundaries is None:
        distro_boundaries = boundaries2[distro] + [(None,None)]
    n = len(ag_distro_init_params)
    distro_init_params = np.array(distro_init_params)
    distro_init_params = np.reshape(distro_init_params,(-1,))
    params = np.concatenate([ag_distro_init_params,distro_init_params])

    obj_fun = objective_age_group_fit(distro, kernel_width, los_cutoff,n)
    result = minimize(
        obj_fun,
        params,
        args=(
            x_train,
            y_train,
        ),
        method=method,
    )
    params = result.x
    ag_distro = params[:n]
    distro_params = params[n:]
    distro_params = np.reshape(distro_params,(n,-1))
    kernels = np.zeros((n,kernel_width),dtype=object)
    for i in range(n):
        kernels[i] = generate_kernel(distro, distro_params[i], kernel_width)
    train_err = obj_fun(result.x, x_train, y_train)
    y_pred = calc_age_convolution(x_test, distro, kernel_width, ag_distro, distro_params, los_cutoff)

    train_length = len(x_train)
    test_err = obj_fun(result.x, x_test[train_length-los_cutoff:], y_test[train_length-los_cutoff:])


    result_dict = {}
    result_dict["params"] = result.x
    result_dict["kernel"] = kernels
    result_dict["curve"] = y_pred
    result_dict["train_error"] = train_err
    result_dict["test_error"] = test_err
    result_dict["minimization_result"] = result
    return result_dict
debug_windows = False
debug_distros = False 



fig,axs = plt.subplots(1+n_ag,1,figsize=(10,7),sharex=True,dpi=150)
for i in range(n_ag):
    axs[i].plot(df_age.values[:,i],color="black",linestyle="--",alpha=.5,label=f"Age Group {i}")
axs[-1].plot(df_age.sum(axis=1).values,color="black",linestyle="--",alpha=.5,label="Total")

for i in range(n_ag):
    axs[i].set_title(df_age.columns[i])
axs[-1].set_title("Total")
plt.xlim(500,1300)
fit_results_by_window = []

trans_rates = []
delay = []

l = list(enumerate(windows))
if debug_windows:
    l = [l[10]]
for i, window in l:
    print("#"*50)
    print(f"Window {i+1}/{len(windows)}")
    print("#"*50)

    w = WindowInfo(window)
    if w.test_end >= len(df):
        continue

    init_transition_rate = 0.014472729492187482
    init_delay = 2



    curve_init_params = [init_transition_rate, init_delay]
    curve_fit_boundaries = [(0, 1),(0,10)]
    if params.use_manual_transition_rate:
        transition_rate = manual_transition_rates[windows[i]]
        curve_init_params = [transition_rate, init_delay]
        curve_fit_boundaries = [
            (transition_rate*.9,transition_rate*1.1),
            (init_delay,init_delay)
        ]
    if params.fit_admissions:
        curve_init_params = [1, 0]
        curve_fit_boundaries = [(1, 1),(0,0)]

    distro_to_fit = list(distributions.keys())
    nono = ["beta","invgauss","gamma"] + ["sentinel","block"]
    distro_to_fit = [distro for distro in distro_to_fit if distro not in nono]
    if debug_distros:
        distro_to_fit = ["linear"]
    fit_results = {}
    for distro_counter,distro in enumerate(distro_to_fit):
        print(f"Fitting {distro} - {distro_counter+1}/{len(distro_to_fit)}")
        if params.use_ag_input_data:
            x_full = df_icu_ag
        else:
            if params.fit_admissions:            
                if params.smooth_data:
                    x_full = df["new_icu_smooth"]
                else:
                    x_full = df["new_icu"]
            else:
                if params.smooth_data:
                    x_full = df["AnzahlFall"]
                else:
                    x_full = df["daily"]
        x_full = x_full.values
        if params.fit_admission_age_groups:
            y_full = df_age.values
        else:
            y_full = df["icu"].values
        x_test = x_full[w.train_start:w.test_end]
        y_test = y_full[w.train_start:w.test_end]

        x_train = x_full[w.train_start:w.train_end]
        y_train = y_full[w.train_start:w.train_end]
        if distro in df_init.index:
            init_values = df_init.loc[distro]["params"]
        else:
            init_values = []
        boundaries = [(val,val) for val in init_values]

        try:

            if params.use_ag_input_data:
                n_ag = x_full.shape[-1]
                result_dicts = [None] * n_ag
                for ag_id in range(n_ag):

                    result_dict = fit_kernel_distro_to_data(
                        distro,
                        x_train[:, ag_id],
                        y_train[:, ag_id],
                        x_test[:, ag_id],
                        y_test[:, ag_id],
                        kernel_width,
                        los_cutoff,
                        curve_init_params,
                        curve_fit_boundaries,
                        distro_init_params=init_values,
                        # distro_boundaries=boundaries,
                        gaussian_spread=params.use_gaussian_spread
                    )
                    result_dicts[ag_id] = result_dict
                observed = [None]*n_ag
                # calc_its_convolution(x_test, fitted_kernel, *result.x[:2], los_cutoff)
                for i in range(n_ag):
                    kernel = result_dicts[ag_id]["kernel"]
                    _params = result_dicts[ag_id]["params"]
                    observed[i] = calc_its_convolution(x_test[:,i],kernel,*_params[:2],los_cutoff)
                    # train
                    xs = np.arange(w.train_start+los_cutoff,w.train_end)
                    sta,sto = los_cutoff,params.train_width
                    # # test
                    # xs = np.arange(w.test_start,w.test_end)
                    # sta, sto = params.train_width, params.train_width + params.test_width
                    # # both
                    # xs = np.arange(w.train_start+los_cutoff,w.test_end)
                    # sta,sto = los_cutoff,len(observed[i])


                    axs[i].plot(xs,observed[i][sta:sto],label=f"{distro.capitalize()} {i}")

                axs[-1].plot(xs,np.sum(observed,axis=0)[sta:sto],label=f"{distro.capitalize()} Total")
                
                result_dict = {}
                result_dict["params"] = np.concatenate([result_dicts[i]["params"] for i in range(n_ag)])
                result_dict["kernel"] = np.stack([result_dicts[i]["kernel"] for i in range(n_ag)])
                result_dict["curve"] = np.stack([result_dicts[i]["curve"] for i in range(n_ag)])
                result_dict["train_error"] = np.mean([result_dicts[i]["train_error"] for i in range(n_ag)])
                result_dict["test_error"] = np.mean([result_dicts[i]["test_error"] for i in range(n_ag)])
                result_dict["minimization_result"] = np.array([result_dicts[i]["minimization_result"] for i in range(n_ag)])
                

            elif params.fit_admission_age_groups:
                
                if "linear" in fit_results:
                    ag_distro_init_params = fit_results["linear"]["params"][:n_ag]
                else:
                    ag_distro_init_params = [1/n_ag] * n_ag
                distro_init_params = np.array([init_values]*n_ag)
                result_dict = fit_age_group_distros(
                   distro,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    kernel_width, 
                    los_cutoff,
                    ag_distro_init_params,
                    distro_init_params,
                    )
                res_params = result_dict["params"]
                observed = calc_age_convolution(x_test,  distro, kernel_width, res_params[:n_ag], np.reshape(res_params[n_ag:],(n_ag,-1)),los_cutoff)
                xs = np.arange(w.train_start+los_cutoff,w.test_end)
                for i in range(n_ag):
                    axs[i].plot(xs,observed[i][los_cutoff:],label=f"{distro.capitalize()} {i}")
                axs[-1].plot(xs,np.sum(observed,axis=0)[los_cutoff:],label=f"{distro.capitalize()} Total")
                y_pred = result_dict['curve'].T
                
                relative_errors = np.abs(y_pred[:len(y_full)]-y_full)/(y_full+1)
                
                
                
            else:
                result_dict = fit_kernel_distro_to_data(
                    distro,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    kernel_width,
                    los_cutoff,
                    curve_init_params,
                    curve_fit_boundaries,
                    distro_init_params=init_values,
                    # distro_boundaries=boundaries,
                    gaussian_spread=params.use_gaussian_spread
                )

                trans_rates.append(result_dict["params"][1])
                delay.append(result_dict["params"][0])
                y_pred = calc_its_convolution(x_full, result_dict["kernel"], *result_dict["params"][:2],los_cutoff,gaussian_spread=params.use_gaussian_spread)
                relative_errors = np.abs(y_pred[:len(y_full)]-y_full)/(y_full+1)
                result_dict["train_relative_error"] = np.mean(relative_errors[w.train_start:w.train_end])
                result_dict["test_relative_error"] = np.mean(relative_errors[w.test_start:w.test_end])
        except Exception as e:
            print(f"\tError in {distro}:",e)
            min_result = types.SimpleNamespace()
            min_result.success = False
            result_dict = {"train_error":np.inf,"test_error":np.inf,"minimization_result":min_result}

        succ = np.all([m.success for m in np.array(result_dict["minimization_result"]).flatten()])
        if not succ:
            # result_dict["train_error"] = np.inf
            # result_dict["test_error"] = np.inf
            print(f"\tFailed to fit {distro}")
        result_dict["success"] = succ
        fit_results[distro] = result_dict

    fit_results_by_window.append(fit_results)
#%%


#%% Generate Video 2
debug = False
if True:
    if not debug:
        path = animation_folder
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        os.makedirs(path)

    window_counter = 1
    stuff = list(zip(windows,fit_results_by_window))
    if debug:
        stuff = [stuff[10]]
    for window, fit_results in stuff:
        print(f"Animation Window {window_counter}/{len(windows)}")
        window_counter+=1
        fig = plt.figure(figsize=(17, 10),dpi=150)
        gs = gridspec.GridSpec(2, 4,height_ratios=[2,1])
        ax1 = fig.add_subplot(gs[0, :4])
        ax11 = ax1.twinx()
        ax2 = fig.add_subplot(gs[1,:2])
        ax3 = fig.add_subplot(gs[1, 2])
        ax32 = fig.add_subplot(gs[1, 3])

        w = WindowInfo(window)

        lines1 = ax1.plot(y_full.sum(axis=1), color="black",label="ICU Bedload")

        span1 = ax1.axvspan(w.train_start, w.train_los_cutoff, color="magenta", alpha=0.1,label=f"Los convolution edge window")
        span2 = ax1.axvspan(w.train_los_cutoff, w.train_end, color="red", alpha=0.2,label=f"Training = {params.train_width-los_cutoff} days")
        span3 = ax1.axvspan(w.test_start, w.test_end, color="blue", alpha=0.05,label=f"Test")
        ax1.axvline(w.train_end,color="black",linestyle="-",linewidth=1)

        label = "COVID Incidence (scaled)"
        if params.fit_admissions:
            label = "New ICU Admissions (scaled)"
        line2 = ax11.plot(x_full,linestyle="--",label=label)
        # use scientific noatation for y axis
        ax11.ticklabel_format(axis="y",style="sci",scilimits=(0,0))
        ma = x_full.max()
        ax11.set_ylim(-ma/7.5,ma*4)

        plot_lines  = []
        replace_names = {"block":"Constant Discharge","sentinel":"Baseline: Sentinel"}
        for i, distro in enumerate(distro_to_fit):
            result = fit_results[distro]
            name = replace_names.get(distro,distro.capitalize())

            if "minimization_result" in result:
                succ = np.all([m.success for m in result_dict["minimization_result"]])
                if not succ:
                    ax2.plot([],[],label=name)
                    l = ax1.plot([],[],label=name)
                    plot_lines.extend(l)
                    continue
            ax2.plot([],[],label=name,color=colors[i])
            ax2.plot(result['kernel'].T,color=colors[i])
            y = result['curve'].T[los_cutoff:]

            s = np.arange(y.shape[0])+los_cutoff+w.train_start
            ls = ax1.plot(s,y.sum(axis=1), label=f"{distro.capitalize()}")
            plot_lines.extend(ls)


        ax2.plot(real_los, color="black",label="Sentinel LoS Charité")

        legend1 = ax1.legend(handles = plot_lines,loc="upper left")
        legend2 = ax1.legend(handles = [*lines1, *line2, span1, span2, span3],loc="upper right")
        ax1.add_artist(legend1)
        ax1.add_artist(legend2)

        ax1.set_title(f"ICU Occupancy")
        ax1.set_xticks(xtick_pos)
        ax1.set_xticklabels(xtick_label)
        if params.fit_admissions:
            ax1.set_xlim(new_icu_day-30,1300)
        else:
            ax1.set_xlim(70,1300)
        ax1.set_ylim(-200,6000)
        ax1.set_ylabel("Occupied Beds")
        ax11.set_ylabel("(Incidence)")
        ax2.legend(loc = "upper right",fancybox=True,ncol=2)
        ax2.set_ylim(0,0.1)
        ax2.set_xlim(-2,80)
        ax2.set_ylabel("Discharge Probability")
        ax2.set_xlabel("Days after admission")
        ax2.set_title(f"Estimated LoS Kernels")


        train_errors = [fit_results[distro]['train_error'] for distro in distro_to_fit]
        test_errors = [fit_results[distro]['test_error'] for distro in distro_to_fit]

        nan_val = -1e5
        train_errors = [nan_val if error == np.inf else error for error in train_errors]
        test_errors =  [nan_val if error == np.inf else error for error in test_errors]

        lim = .5e7
        for i, distro in enumerate(distro_to_fit):
            patch, = ax3.bar([distro],train_errors[i],label="Train",color=colors[i])
            c = patch.get_facecolor()
            ax32.bar([distro],test_errors[i],label="Test",color=c)
            w = .6

            if fit_results[distro]["success"]:
                color = "green"
            else:
                color = "red"
            rect = plt.Rectangle(xy=(-w/2 + i,lim*.9),width=w,height=lim*.08,color=color,alpha=.55)
            ax3.add_patch(rect)



        ax3.set_ylim(0,lim)
        ax3.set_title("Train Error")
        ax3.set_xticks(distro_to_fit)
        ax3.set_xticklabels(distro_to_fit,rotation=75)
        ax3.set_ylabel("MSE")

        ax32.set_ylim(0,lim)
        ax32.set_title("Test Error")
        ax32.set_xticks(distro_to_fit)
        ax32.set_xticklabels(distro_to_fit,rotation=75)
        ax32.set_ylabel("MSE")

        plt.suptitle(f"Deconvolution Training Process\n{run_name}",fontsize=16)
        plt.tight_layout()
        if debug:
            show_plt()
        else:
            plt.savefig(animation_folder + f"fit_{window:04d}.png")
            plt.close()
        plt.clf()
#%%
# plot age groups
fig,(ax1,*axs) = plt.subplots(5,1, figsize=(10,7), sharex=True)
ax1.plot(y_full, label="ICU Bedload")
ax1.legend(df_age.columns)
ags = list()
for i in range(len(windows)):
    fit_results = fit_results_by_window[i]
    p = fit_results["linear"]["params"][:2]
    ags.append([*p,0])
ags = np.array(ags)

ags[:,2] = 1-ags.sum(axis=1)
axs[0].plot(windows,ags[:,0])
axs[1].plot(windows,ags[:,1])
axs[2].plot(windows,ags[:,2])
for ax in axs:
    ax.set_ylim(0,1)
axs[3].plot(windows,ags[:,0])
axs[3].plot(windows,ags[:,1])
axs[3].plot(windows,ags[:,2])

ax1.set_xlim(550,1400)
#%%
#Save results
import pickle
with open(results_folder + "fit_results.pkl","wb") as f:
    pickle.dump(fit_results_by_window,f)

#%%
# Plot number of failed fits and successfull fits
n_success = []
for distro in distro_to_fit:
    n = sum([1 for fit_results in fit_results_by_window if fit_results[distro]["minimization_result"].success == True])
    n_success.append(n)
plt.figure(figsize=(10,5),dpi=150)
plt.bar(distro_to_fit,n_success)
plt.title("Number of successful fits")
plt.axhline(len(fit_results_by_window),color="red",linestyle="--",label="Total")
plt.xticks(rotation=45)
plt.savefig(figures_folder + "successful_fits.png")

show_plt()

#%%
# Calculate success rate
train_errors_by_distro = [[fit[distro]['train_error'] for fit in fit_results_by_window] for distro in distro_to_fit]
test_errors_by_distro = [[fit[distro]['test_error'] for fit in fit_results_by_window] for distro in distro_to_fit]
success_by_distro = [[fit[distro]["minimization_result"].success for fit in fit_results_by_window] for distro in distro_to_fit]
failure_by_distro = [[0 if success else 1 for success in successes] for successes in success_by_distro]

# Convert losses to DataFrame
df_train = pd.DataFrame(np.array(train_errors_by_distro).T, columns=distro_to_fit)
df_test = pd.DataFrame(np.array(test_errors_by_distro).T, columns=distro_to_fit)

df_failures = pd.DataFrame(np.array(failure_by_distro).T, columns=distro_to_fit)
# Compute mean finite loss and failure rate for each model
summary = pd.DataFrame(index=distro_to_fit)
summary["Mean Loss Train"] = df_train.replace(np.inf, np.nan).mean()
summary["Median Loss Train"] = df_train.replace(np.inf, np.nan).median()
summary["Failure Rate Train"] = df_failures.mean()
summary["Upper Quartile Train"] = df_train.quantile(0.75)
summary["Lower Quartile Train"] = df_train.quantile(0.25)

summary["Mean Loss Test"] = df_test.replace(np.inf, np.nan).mean()
summary["Median Loss Test"] = df_test.replace(np.inf, np.nan).median()
summary["Failure Rate Test"] = df_failures.mean()
# add column for mean loss without oultiers
col = "Mean Loss Test (no outliers)"
summary[col] = np.nan

#%%
# Find outliers
for distro in distro_to_fit:
    Q1 = df_test[distro].quantile(0.25)
    Q3 = df_test[distro].quantile(0.75)
    IQR = Q3 - Q1
    # filter out outliers
    mask = (df_test[distro] < (Q1 - 1.5 * IQR)) | (df_test[distro] > (Q3 + 1.5 * IQR))
    summary.at[distro,col] = df_test[distro][~mask].mean()

#%%
# save test errors and failure rates in csv
df_test.to_csv(results_folder + "test_errors.csv")
df_failures.to_csv(results_folder + f"failure_rates.csv")
summary.to_csv(results_folder + "summary.csv")

#%%

# Visualization
def viz(col2, col1,ylim=None,save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6),dpi=150)

    for i, distro in enumerate(distro_to_fit):
        if distro in ["sentinel","block"]:
            continue
        val1 = summary[col1][distro]
        val2 = summary[col2][distro]
        ax.scatter(val1, val2, s=100, label=distro, color=colors[i])
        ax.annotate(distro, (val1, val2), fontsize=9, xytext=(5,5), textcoords='offset points')
    # ax.set_yscale("log")

    # Labels and formatting
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_title(f"Model Performance: {col1} vs. {col2}\n{run_name}")
    if ylim is not None:
        ax.set_ylim(ylim)
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path)
    show_plt()
viz("Median Loss Train", "Failure Rate Train",          save_path=figures_folder +  "median_loss_vs_failure_rate_train.png")
viz("Median Loss Test", "Failure Rate Test",save_path=figures_folder +  "median_loss_vs_failure_rate_test.png")
viz(col, "Failure Rate Test",save_path=figures_folder +  "mean_loss_vs_failure_rate_test.png")
#%%
# sorted summary
sorted_summary = summary.sort_values("Median Loss Test")
sorted_summary = sorted_summary[["Median Loss Test","Failure Rate Train","Median Loss Train","Upper Quartile Train","Lower Quartile Train"]]
sorted_summary.plot(subplots=True,figsize=(10,10))
plt.legend()
plt.title("Median Loss")
xticks = list(sorted_summary.index)
plt.xticks(np.arange(len(xticks)),xticks,rotation=45)
show_plt()
#%%
#remove inf from both
train_errors_by_distro = [[error for error in errors if error != np.inf] for errors in train_errors_by_distro]
test_errors_by_distro = [[error for error in errors if error != np.inf] for errors in test_errors_by_distro]
#%%

plt.figure(figsize=(10,5),dpi=150)
plt.boxplot(train_errors_by_distro)
distro_and_n = [f"{distro.capitalize()} n={succs}" for distro,succs in zip(distro_to_fit,n_success)]
plt.xticks(np.arange(len(distro_to_fit))+1,distro_and_n,rotation=45)
plt.title("Train Error - Logarithmic")
plt.yscale("log")
plt.ylabel("MSE - Logarithmic Scale")
plt.tight_layout()
plt.savefig(figures_folder + "train_error_boxplot.png")
show_plt()

plt.figure(figsize=(10,5),dpi=150)
plt.boxplot(test_errors_by_distro)
plt.xticks(np.arange(len(distro_to_fit))+1,distro_and_n,rotation=45)
plt.title("Test Error - Logarithmic")
plt.yscale("log")
plt.ylabel("MSE - Logarithmic Scale")
plt.tight_layout()
plt.savefig(figures_folder + "test_error_boxplot.png")
show_plt()
#%%
fig = plt.figure(figsize=(10,5))
sns.stripplot(data=train_errors_by_distro, jitter=0.2)
plt.xticks(np.arange(len(distro_to_fit)),distro_to_fit,rotation=45)
plt.title("Train Error - Logarithmic Scale")
plt.yscale("log")
plt.savefig(figures_folder + "train_error_stripplot.png")
show_plt()



#%%
for distro in distro_to_fit:

    fig,(ax,ax4,ax2)= plt.subplots(3,1,figsize=(10,5),sharex=True,dpi=150)
    ax.plot(y_full, color="black",label="Real" ,alpha=.8,linestyle="--")
    for i,fit_results in enumerate(fit_results_by_window):
        window = windows[i]
        if fit_results[distro]["minimization_result"].success == False:
            start, end = window-params.train_width, window
            ax.axvspan(start,end, color="red", alpha=0.1)
            continue
        result = fit_results[distro]
        y = result['curve']
        w = WindowInfo(window)

        _y = y[los_cutoff:params.train_width]
        ax.plot(
            np.arange(w.train_los_cutoff,w.train_end)[:len(_y)],
            _y,
            color=colors[0],
        )
        _y = y[params.train_width:params.train_width+params.test_width]
        ax.plot(
            np.arange(w.train_end,w.test_end)[:len(_y)],
            _y,
            color=colors[1],
        )


    ax.plot([],[],color=colors[0],label = f"{distro.capitalize()} Train")
    ax.plot([],[],color=colors[1],label = f"{distro.capitalize()} Prediction")
    ax.axvspan(0,0, color="red", alpha=0.1,label="Failed Training Windows")
    ax.axvspan(sentinel_start_day,sentinel_end_day, color="green", alpha=0.1,label="Sentinel Window")
    ax.legend(loc="upper left")
    ax.set_ylim(-100,6000)
    ax.set_xticks(xtick_pos[::2])
    ax.set_xticklabels(xtick_label[::2])
    if params.fit_admissions:
        ax.set_xlim(new_icu_day-80,1300)
    else:
        ax.set_xlim(50,1300)
    ax.grid()

    trans_probs = np.zeros(len(fit_results_by_window))*np.nan
    trans_delay = np.zeros(len(fit_results_by_window))*np.nan
    _train_errs = np.zeros(len(fit_results_by_window))*np.nan
    _test_errs = np.zeros(len(fit_results_by_window))*np.nan
    for i,fit_results in enumerate(fit_results_by_window):
        _train_errs[i] = fit_results[distro]["train_error"]
        _test_errs[i] = fit_results[distro]["test_error"]

        if fit_results[distro]["minimization_result"].success == False:
            continue
        trans_probs[i] = fit_results[distro]["params"][0]
        trans_delay[i] = fit_results[distro]["params"][1]

    # plot trans rates in ax2
    ax2.bar(windows,trans_probs, width=15 ,label="Transition Probability")
    ax2.grid()
    ax2.set_ylim(-.01,0.1)
    ax2.set_title("Transition Probability")

    ax4.plot(windows,_train_errs,label = "Train Error")
    ax4.plot(windows,_test_errs,label = "Test Error")
    # mark nan and inf values
    for i in range(len(windows)):
        if fit_results_by_window[i][distro]["minimization_result"].success == False:
            ax4.axvline(windows[i],color="red",alpha=.5)
    ax4.axvline(-np.inf,color="red",label="Failed Fit",alpha=.5)
    # ax4.axvspan(sentinel_start_day,sentinel_end_day, color="green", alpha=0.1,label="Sentinel Window")
    ax4.legend(loc="upper right")
    ax4.set_ylim(-1e4,1e5)
    ax4.set_title("Error")
    ax4.grid()

    plt.suptitle(f"{distro.capitalize()} Distribution")
    plt.tight_layout()
    plt.savefig(figures_folder + f"prediction_error_{distro}_fit.png")
    show_plt()
#%%
fig,(ax,ax4)= plt.subplots(2,1,figsize=(12,6),sharex=True,dpi=300)
ax.plot(y_full, color="black",label="Real" ,alpha=.8,linestyle="--")
ax.plot([],[],color=colors[0],label = f"{distro.capitalize()} Train")
ax.plot([],[],color=colors[1],label = f"{distro.capitalize()} Prediction")
ax.axvspan(0,0, color="red", alpha=0.1,label="Failed Training Windows")
for distro in distro_to_fit:

    for i,fit_results in enumerate(fit_results_by_window):
        window = windows[i]
        if fit_results[distro]["minimization_result"].success == False:
            start, end = window-params.train_width, window
            ax.axvspan(start,end, color="red", alpha=0.01)
            continue

        result = fit_results[distro]
        y = result['curve']
        w = WindowInfo(window)

        _y = y[los_cutoff:params.train_width]
        l1, = ax.plot(
            np.arange(w.train_los_cutoff,w.train_end)[:len(_y)],
            _y,
            color=colors[0],
        )
        _y = y[params.train_width:params.train_width+params.test_width]
        l2, = ax.plot(
            np.arange(w.train_end,w.test_end)[:len(_y)],
            _y,
            color=colors[1],
        )




    trans_probs = np.zeros(len(fit_results_by_window))*np.nan
    trans_delay = np.zeros(len(fit_results_by_window))*np.nan
    _train_errs = np.zeros(len(fit_results_by_window))*np.nan
    _test_errs = np.zeros(len(fit_results_by_window))*np.nan
    for i,fit_results in enumerate(fit_results_by_window):
        _train_errs[i] = fit_results[distro]["train_error"]
        _test_errs[i] = fit_results[distro]["test_error"]

        if fit_results[distro]["minimization_result"].success == False:
            continue
        trans_probs[i] = fit_results[distro]["params"][0]
        trans_delay[i] = fit_results[distro]["params"][1]




    ax4.plot(windows,_train_errs,color=colors[0])
    ax4.plot(windows,_test_errs, color=colors[1])

ax.set_ylim(-100,6000)
if params.fit_admissions:
    ax.set_xlim(new_icu_day-80,1300)
ax.grid()
ax.legend()
ax4.axvline(-np.inf,color="red",label="Failed Fit",alpha=.5)
ax4.legend(["Train Errors","Test Errors"],loc="upper right")
ax4.set_ylim(-1e4,1e5)
ax4.grid()
# ax4.set_xticks(xtick_pos,xtick_label)

ax4.set_xlabel("Time")
ax4.set_ylabel("Error")
ax.set_ylabel("ICU")
# set xticks
ax.set_xticks(xtick_pos[::2])
ax.set_xticklabels(xtick_label[::2])
if params.fit_admissions:
    ax.set_xlim(new_icu_day-30,1300)
plt.savefig(figures_folder + "prediction_error_all_distros.png")
show_plt()

#%%
fig,ax= plt.subplots(1,1,figsize=(15,7.5),sharex=True)
ax.plot(y_full, color="black",label="Real" ,alpha=.8,linestyle="--")

for distro in distro_to_fit:
    for i,fit_results in enumerate(fit_results_by_window):
        if fit_results[distro]["minimization_result"].success == False:
            continue
        window = windows[i]
        w = WindowInfo(window)
        result = fit_results[distro]
        y = result['curve']

        _y = y[los_cutoff:params.train_width]
        ax.plot(
            np.arange(w.train_los_cutoff,w.train_end)[:len(_y)],
            _y,
            color=colors[0],
        )
        _y = y[params.train_width:params.train_width+params.test_width]
        ax.plot(
            np.arange(w.train_end,w.test_end)[:len(_y)],
            _y,
            color=colors[1],
        )

ax.set_ylim(-100,6000)
if params.fit_admissions:
    ax.set_xlim(new_icu_day-30,1300)
ax.grid()

ax.plot([],[],color=colors[0],label = f"All Distros Train")
ax.plot([],[],color=colors[1],label = f"All Distros Prediction")
ax.legend()
plt.savefig(figures_folder + "prediction_error_all_fits.png")
show_plt()


# %%
# plot distributions

for distro in distro_to_fit:
    fig, ax = plt.subplots(figsize=(10,5))
    # plot real kernel
    ax.plot(real_los,color='black',label="Real")

    for result in fit_results_by_window:
        if result[distro]["minimization_result"].success == False:
            continue
        y = result[distro]["kernel"]
        ax.plot(y,alpha=0.3,color=colors[0])

    plt.grid()
    plt.legend()
    plt.title(f"{distro.capitalize()} Kernel")
    plt.ylim(-0.005,0.3)
    plt.tight_layout()
    plt.savefig(figures_folder + f"all_kernels_{distro}.png")
    show_plt()
#%%
good_distros = ["exponential","gaussian","t","block","sentinel"]

#%%

if "sentinel" in distro_to_fit:
    from scipy.optimize import minimize

    # Fit transition rates manually
    tr = np.array([fit_results["sentinel"]["params"][0] for fit_results in fit_results_by_window])
    tr = np.insert(tr,0,tr[0])


    # Fit transition rates manually
    tr = np.array([fit_results["sentinel"]["params"][0] for fit_results in fit_results_by_window])
    tr = np.insert(tr,0,tr[0])
    mps = np.array([
        [0,0.001],
        [48,0.001],
        [69,0.001],
        [111,0.05146966],
        [174,0.05982728],
        [237,0.01350479],
        [500,0.01350479],
        [594,0.01850925],
        [783,0.00042575],
        [1119,0.0017396 ],
        [1329,0.02537781],
    ])
    xs = np.arange(len(df))
    windows2 = np.insert(windows.copy(),0,0)
    tr_y = np.interp(xs,windows2,tr)

    def obj_fun(ps):
        ps = ps.reshape(-1,2)
        ys = np.interp(xs,ps[:,0],ps[:,1])
        return np.mean((ys-tr_y)**2)
    result = minimize(
        obj_fun,
        mps.flatten(),
        method = "L-BFGS-B"
        )
    print(result)

    fig,ax = plt.subplots(1,1,figsize=(10,5))
    res = result.x.reshape(-1,2)
    ys = np.interp(xs,res[:,0],res[:,1])
    plt.plot(ys,label="Fitted Transition Rates")
    plt.plot(tr_y,label="Real Transition Rates")
    plt.plot(np.abs(ys-tr_y),label="Difference")
    plt.grid()
    plt.legend()
    show_plt()
    manual_points = np.array(
        [[0,               5.26125540e-02],
        [ 4.79999995e+01,  5.29817366e-02],
        [ 6.89999731e+01,  5.25110807e-02],
        [ 1.10999986e+02,  5.42751008e-02],
        [ 1.73999994e+02,  3.54917868e-02],
        [ 2.37000003e+02,  1.42336509e-02],
        [ 4.99999999e+02,  1.48902533e-02],
        [ 5.94000000e+02,  1.08504095e-02],
        [ 7.83000001e+02,  1.49297704e-03],
        [ 1.11900000e+03,  8.93606945e-04],
        [ 1.32900000e+03,  1.76263698e-02]]
    )
    xs = np.arange(len(df))
    ys = np.interp(xs,manual_points[:,0],manual_points[:,1])

    print(obj_fun(result.x))
    print(result.x.reshape(-1,2))
# %%
transition_rates = np.array([[fit_results[distro]["params"][0] for fit_results in fit_results_by_window] for distro in distro_to_fit])

# transition rates in a subplot together with the icu
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,10),sharex=True)
# ax1.plot(df["icu"].values,label="ICU")
y = df["icu"].values/df["AnzahlFall"].values
ax1.plot(y,label="icu incidence ratio")
ax1.set_title("ICU Occupancy")
ax1.legend()
dddd =['lognorm',
'weibull',
'gaussian',
'exponential',
'gamma',
'beta',
'cauchy',
't',
'invgauss',
'block'
]
nono=[
    "weibull",
    "block"
]
for i,distro in enumerate(distro_to_fit):
    if distro in nono:
        continue
    ax2.plot(windows,transition_rates[i],label=distro)
ax2.axhline(0,color="black",linestyle="--")
ax2.legend()
ax2.set_title("Transition Rates")
# ax2.set_ylim(-.03,0.2)
plt.savefig(figures_folder + "transition_rates_and_icu.png")
show_plt()
#%%
# Generate ensemble of predictions
all_predictions_combined = []
for i, fit_results in enumerate(fit_results_by_window):
    window = windows[i]
    pred = np.zeros((len(df),len(distro_to_fit)))*np.nan
    for window_counter,distro in enumerate(distro_to_fit):
        if fit_results[distro]["minimization_result"].success == False:
            continue
        result = fit_results[distro]
        y = result['curve']
        w = WindowInfo(window)
        _y = y[params.train_width:params.train_width+params.test_width]
        pred[w.train_end:w.test_end,window_counter] = _y
    all_predictions_combined.append(pred)
all_predictions_combined = np.array(all_predictions_combined)
all_predictions_combined.shape # in shape (n_windows, n_days, n_distros)

#%%
distros = []
for i, distro in enumerate(distro_to_fit):
    if distro in ["cauchy","exponential","t","gaussian","sentinel"]:
        distros.append((i,distro))
fig,(ax,ax2) = plt.subplots(2,1,figsize=(10,5),sharex=True,dpi=150)
# plot icu and incidence
ax.plot(y_full, color="black",label="ICU Bedload")
for i, window in enumerate(windows):
    w = WindowInfo(window)
    x = np.arange(w.train_end,w.test_end)
    for j, distro in distros:
        y = all_predictions_combined[i,w.train_end:w.test_end,j]
        ax.plot(x,y,color=colors[j],alpha=1)
ax.set_ylim(-100,6000)
ax.set_title("All Predictions")
ax.legend()

transition_rates = np.array([[fit_results[distro]["params"][0] for fit_results in fit_results_by_window] for distro in distro_to_fit])
lines =[]
for i,distro in distros:
    l, = ax2.plot(windows, transition_rates[i],label=distro)
    lines.append(l)
ax2.legend(handles=lines,ncol=2,loc="upper right")
ax2.set_xticks(xtick_pos[::2])
ax2.set_xticklabels(xtick_label[::2])
# ax2.set_ylim(-.03,1.03)
ax2.set_ylim(-.01,.075)
ax2.set_xlim(50,1300)
ax2.set_title("Transition Rates")
plt.tight_layout()
plt.savefig(figures_folder + "transition_rates.png")
show_plt()
#%%
# plot all errors in a graph
fig,ax = plt.subplots(1,1,figsize=(10,5),dpi=150)
for i, distro in enumerate(distro_to_fit):
    train_errors = [fit_results[distro]["train_error"] for fit_results in fit_results_by_window]
    test_errors = [fit_results[distro]["test_error"] for fit_results in fit_results_by_window]
    ax.plot(windows,train_errors,label=f"{distro.capitalize()} Train")
    ax.plot(windows,test_errors,label=f"{distro.capitalize()} Test")
ax.legend()
ax.set_title("All Errors")
ax.set_xticks(xtick_pos[::2])
ax.set_xticklabels(xtick_label[::2])
plt.grid()

#%%
os.system('msg * "Finished all calculations"')


# %%
# calculate 90% containment for all kernels
plt.figure(figsize=(10,5),dpi=150)
cutoffs = []
for distro in distro_to_fit:
    if distro in ["block","sentinel"]:
        continue
    for result in fit_results_by_window:
        if result[distro]["minimization_result"].success == False:
            continue
        y = result[distro]["kernel"]
        # find the point, where 90% of the probability mass is contained
        cumsum = np.cumsum(y)
        idx = np.argmax(cumsum > 0.99)
        cutoffs.append(idx)
        plt.plot(y[:idx],alpha=0.1)
plt.title("90% Containment")
plt.grid()
plt.show()
plt.plot(sorted(cutoffs))
plt.axvline(len(cutoffs)//2,color="red",linestyle="--")
plt.axvline(len(cutoffs)//4*3,color="red",linestyle="--")
plt.plot()
#%%