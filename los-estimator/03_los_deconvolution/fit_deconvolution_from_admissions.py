#%%
%load_ext autoreload
%autoreload 2
import sys
import numpy as np
import pandas as pd
import types
import seaborn as sns


from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
sys.path.append("../02_fit_los_distributions/")
from dataprep import load_los, load_incidences, load_icu_occupancy
from los_fitter import fit_kernel_distro_to_data, distributions, calc_its_convolution
plt.rcParams['savefig.facecolor']='white'
#%%
los_file = "../01_create_los_profiles/berlin/output_los/los_berlin_all.csv"

kernel_width = 120
los_cutoff = 41

initial_guess_prob = 0.018


start_day = "2020-01-01"
end_day = "2025-01-01"

# train_start_day = "2020-07-01"
# train_end_day = "2020-12-31"

train_start_day = "2020-07-01"
train_end_day = "2020-12-31"

def date_to_day(date):
    return (date - pd.Timestamp(start_day)).days
def day_to_date(day):
    return pd.Timestamp(start_day) + pd.Timedelta(days=day)
train_start = date_to_day(pd.Timestamp(train_start_day))
train_end = date_to_day(pd.Timestamp(train_end_day))

#%%
# take matplotlib standart color wheel
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# add extra color palette
colors += ["#FFA07A","#20B2AA","#FF6347","#808000","#FF00FF","#FFD700","#00FF00","#00FFFF","#0000FF","#8A2BE2"]

#%%
real_los, _ = load_los(file=los_file)
df_inc = load_incidences(start_day, end_day)
df_icu = load_icu_occupancy(start_day, end_day)
df_icu["new_icu"] = df_icu["new_icu"].rolling(7).mean()
df = df_inc.join(df_icu,how="inner")

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
train_width = 100
test_width = 21
step = 21
windows = np.arange(new_icu_day,len(df)-step,step)
# remove windows, where y_test is too short
windows = np.array([window for window in windows if window+train_width+test_width < len(df)])+train_width


#%%
# Fitting the LoS curves, as well as the delay and probability

from los_fitter import fit_kernel_distro_to_data,distributions
import os

fit_results_by_window = []


l = list(enumerate(windows))
for i, window in l: # [l[10]]:
    print("#"*50)
    print(f"Window {i+1}/{len(windows)}")
    print("#"*50)

    train_start = window-train_width
    train_end = window
    test_start = train_end
    test_end = test_start + test_width
    if test_end >= len(df):
        continue    

    init_transition_rate = 1
    init_delay = 0

    curve_init_params = [init_transition_rate, init_delay]
    curve_fit_boundaries = [(0, 2),(0,0)]
    # curve_fit_boundaries= [(init_transition_rate,init_transition_rate),(init_delay,init_delay)]

    distros = list(distributions.keys()) 
    fit_results = {}
    for j,distro in enumerate(distros):
        print(f"Fitting {distro} - {j+1}/{len(distros)}")
        x_full = df["new_icu"]
        y_full = df["icu"].values
        x_test = x_full[train_start:test_end]
        y_test = y_full[train_start:test_end]

        x_train = x_full[train_start:train_end]
        y_train = y_full[train_start:train_end]
        if distro in df_init.index:
            init_values = df_init.loc[distro]["params"]
        else:
            init_values = []
        boundaries = [(val,val) for val in init_values]
        try:
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
                # distro_boundaries=boundaries
                )
            y_pred = calc_its_convolution(x_full, result_dict["kernel"], *result_dict["params"][:2],los_cutoff)
            relative_errors = np.abs(y_pred[:len(y_full)]-y_full)/y_full
            result_dict["train_relative_error"] = np.mean(relative_errors[train_start:train_end])
            result_dict["test_relative_error"] = np.mean(relative_errors[test_start:test_end])
        except Exception as e:
            print(f"\tError in {distro}:",e)
            min_result = types.SimpleNamespace()
            min_result.success = False
            result_dict = {"train_error":np.inf,"test_error":np.inf,"minimization_result":min_result}

        if result_dict["minimization_result"].success == False:
            # result_dict["train_error"] = np.inf
            # result_dict["test_error"] = np.inf
            print(f"\tFailed to fit {distro}")
        result_dict["success"] = result_dict["minimization_result"].success
        fit_results[distro] = result_dict

    fit_results_by_window.append(fit_results)
raw_fit_results = fit_results_by_window
raw_distros = distros
print("ok.")
#%%
nono = [
    # 'cauchy',
    # 'invgauss',
    'block',
    'sentinel'
]
fit_results_by_window = [dict() for _ in range(len(raw_fit_results))]
distros = []
for i, distro in enumerate(raw_distros):
    if distro in nono:
        continue
    distros.append(distro)
    for i, fit_results in enumerate(raw_fit_results):
        fit_results_by_window[i][distro] = fit_results[distro]


#%%
train_errors_by_distro = [[fit[distro]['train_error'] for fit in fit_results_by_window] for distro in distros]
test_errors_by_distro = [[fit[distro]['test_error'] for fit in fit_results_by_window] for distro in distros]
success_by_distro = [[fit[distro]["minimization_result"].success for fit in fit_results_by_window] for distro in distros]
# replace by 1
failure_by_distro = [[0 if success else 1 for success in successes] for successes in success_by_distro]

# Convert losses to DataFrame
df_train = pd.DataFrame(np.array(train_errors_by_distro).T, columns=distros)
df_test = pd.DataFrame(np.array(test_errors_by_distro).T, columns=distros)

df_failures = pd.DataFrame(np.array(failure_by_distro).T, columns=distros)
# Compute mean finite loss and failure rate for each model
summary = pd.DataFrame(index=distros)
summary["Mean Loss Train"] = df_train.replace(np.inf, np.nan).mean()
summary["Median Loss Train"] = df_train.replace(np.inf, np.nan).median()
summary["Failure Rate Train"] = df_failures.mean()
summary["Upper Quartile Train"] = df_train.quantile(0.75)
summary["Lower Quartile Train"] = df_train.quantile(0.25)

summary["Mean Loss Test"] = df_test.replace(np.inf, np.nan).mean()
summary["Median Loss Test"] = df_test.replace(np.inf, np.nan).median()
summary["Failure Rate Test"] = df_failures.mean()


# Visualization
def viz(col1, col2,distros = list(enumerate(distros)),save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, distro in distros:
        if distro =="gamma":
            continue
        val1 = summary[col1][distro]
        val2 = summary[col2][distro]
        ax.scatter(val1, val2, s=100, label=distro, color=colors[i])
        ax.annotate(distro, (val1, val2), fontsize=9, xytext=(5,5), textcoords='offset points')

    # Labels and formatting
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.set_title(f"Model Performance: {col1} vs. {col2}")
        
    # plt.xlim(1e6, 1.2e6)
    plt.ylim(-.05,1.05)
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()
viz("Median Loss Train", "Failure Rate Train",save_path="./figures_new_icu/median_loss_vs_failure_rate_train.png")
viz("Median Loss Test", "Failure Rate Test",save_path="./figures_new_icu/median_loss_vs_failure_rate_test.png")


#%%
fig,ax= plt.subplots(1,1,figsize=(15,7.5),sharex=True)
ax.plot(y_full, color="black",label="Real" ,alpha=.8,linestyle="--")

for i,distro in enumerate(distros):
    for i,fit_results in enumerate(fit_results_by_window):    
        if fit_results[distro]["minimization_result"].success == False:
            continue
        window = windows[i]
        result = fit_results[distro]
        y = result['curve']
        train_start = window+41-train_width   
        train_end = window
        test_end = train_end + test_width
        
        _y = y[41:train_width] 
        l1, = ax.plot(
            np.arange(train_start,train_end)[:len(_y)],
            _y,
            color=colors[0],
        )
        _y = y[train_width:train_width+test_width]
        l2, = ax.plot(
            np.arange(train_end,test_end)[:len(_y)],
            _y,
            color=colors[1],
        )
    
ax.set_ylim(-100,6000)
ax.grid()

ax.plot([],[],color=colors[0],label = f"All Distros Train")
ax.plot([],[],color=colors[1],label = f"All Distros Prediction")
ax.legend()

plt.savefig(f"./figures_new_icu/prediction_error_all_fits.png")
plt.show()
#%%
# Plot number of failed fits and successfull fits
n_success = []
for i, distro in enumerate(distros):
    n = sum([1 for fit_results in fit_results_by_window if fit_results[distro]["minimization_result"].success == True])
    n_success.append(n)
plt.bar(distros,n_success)
plt.title("Number of successful fits")
plt.axhline(len(fit_results_by_window),color="red",linestyle="--",label="Total")
plt.xticks(rotation=45)
plt.savefig("./figures_new_icu/successful_fits.png")

plt.show()

#%%
display(summary)
#%%
# sorted summary
sorted_summary = summary.sort_values("Median Loss Test")
sorted_summary = sorted_summary[["Median Loss Test","Failure Rate Train","Median Loss Train","Upper Quartile Train","Lower Quartile Train"]]
sorted_summary.plot(subplots=True,figsize=(10,10))
plt.legend()
plt.title("Median Loss")
xticks = list(sorted_summary.index)
plt.xticks(np.arange(len(xticks)),xticks,rotation=45)
plt.show()
#%%
import seaborn as sns
summary_temp = summary[["Median Loss Test","Failure Rate Train","Median Loss Train"]]
sns.pairplot(summary_temp)
# plt.suptitle("Pairplot of Median Loss Test, Failure Rate Train and Median Loss Train")
# Add Title on top of the figure
plt.subplots_adjust(top=0.95)

plt.suptitle("Comparison of Training Results")
plt.savefig("./figures_new_icu/pairplot.png")

plt.show()
#%%
#remove inf from both
train_errors_by_distro = [[error for error in errors if error != np.inf] for errors in train_errors_by_distro]
test_errors_by_distro = [[error for error in errors if error != np.inf] for errors in test_errors_by_distro]

#%%
train_medians = [np.median(errors) for errors in train_errors_by_distro]
test_medians = [np.median(errors) for errors in test_errors_by_distro]
train_means = [np.mean(errors) for errors in train_errors_by_distro]
test_means = [np.mean(errors) for errors in test_errors_by_distro]
train_percentile_25 = [np.percentile(errors,25) for errors in train_errors_by_distro]
test_percentile_25 = [np.percentile(errors,25) for errors in test_errors_by_distro]
train_percentile_75 = [np.percentile(errors,75) for errors in train_errors_by_distro]
test_percentile_75 = [np.percentile(errors,75) for errors in test_errors_by_distro]
# plot train and test errors per distro and error bars
plt.plot(distros, train_medians,label="Median Train Error")
plt.plot(distros, test_medians,label="Median Test Error")

plt.fill_between(distros, train_percentile_25,train_percentile_75,alpha=0.3)
plt.fill_between(distros, test_percentile_25,test_percentile_75,alpha=0.3)
plt.xticks(rotation=45)
plt.legend()
plt.title("Median Train and Test Error")
plt.show()
#%%

plt.boxplot(train_errors_by_distro)
distro_and_n = [f"{distro.capitalize()} n={succs}" for distro,succs in zip(distros,n_success)]
plt.xticks(np.arange(len(distros))+1,distro_and_n,rotation=45)
plt.title("Train Error")
plt.savefig("./figures_new_icu/train_error_boxplot.png")
plt.show()

plt.boxplot(test_errors_by_distro)
plt.xticks(np.arange(len(distros))+1,distro_and_n,rotation=45)
plt.title("Test Error")
plt.savefig("./figures_new_icu/test_error_boxplot.png")
plt.show()
#%%
fig = plt.figure(figsize=(10,5))
sns.stripplot(data=train_errors_by_distro, jitter=0.2)
plt.xticks(np.arange(len(distros)),distros,rotation=45)
plt.title("Train Error")
plt.show()




#%%
s = 1
fig,(ax,ax4)= plt.subplots(2,1,figsize=(10*s,5*s),sharex=True)
plt.show()
for distro in distros:
    fig,(ax,ax4)= plt.subplots(2,1,figsize=(10*s,5*s),sharex=True)
    ax.plot(y_full, color="black",label="Real" ,alpha=.8,linestyle="--")
    for i,fit_results in enumerate(fit_results_by_window):    
        if fit_results[distro]["minimization_result"].success == False:
            start, end = windows[i]-train_width, windows[i]
            ax.axvspan(start,end, color="red", alpha=0.1)
            continue
        window = windows[i]
        result = fit_results[distro]
        y = result['curve']
        train_start = window+41-train_width
        train_end = window
        test_end = train_end + test_width
        
        _y = y[41:train_width] 
        l1, = ax.plot(
            np.arange(train_start,train_end)[:len(_y)],
            _y,
            color=colors[0],
        )
        _y = y[train_width:train_width+test_width]
        l2, = ax.plot(
            np.arange(train_end,test_end)[:len(_y)],
            _y,
            color=colors[1],
        )
    
    ax.set_ylim(-100,6000)
    ax.grid()

    ax.plot([],[],color=colors[0],label = f"{distro.capitalize()} Train")
    ax.plot([],[],color=colors[1],label = f"{distro.capitalize()} Prediction")
    ax.axvspan(0,0, color="red", alpha=0.1,label="Failed Training Windows")
    ax.axvspan(sentinel_start_day,sentinel_end_day, color="green", alpha=0.1,label="Sentinel Window")
    ax.legend()
    # ax.legend([l1,l2],[f"{distro.capitalize()} Train",f"{distro.capitalize()} Prediction"],loc="upper right")


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


    # prob_line, = ax2.plot(windows,trans_probs,label = "Hospitalization Probability")
    # delay_line, = ax3.plot(windows,trans_delay,label = "Hospitalization Delay",color=colors[1])
    # ax2.legend([prob_line,delay_line],["Hospitalization Probability","Hospitalization Delay"],loc="upper right")


    ax4.plot(windows,_train_errs,label = "Train Error")
    ax4.plot(windows,_test_errs,label = "Test Error")
    # mark nan and inf values
    for i in range(len(windows)):
        if fit_results_by_window[i][distro]["minimization_result"].success == False:
            ax4.axvline(windows[i],color="red",alpha=.5)
    ax4.axvline(-np.inf,color="red",label="Failed Fit",alpha=.5)
    ax4.axvspan(sentinel_start_day,sentinel_end_day, color="green", alpha=0.1,label="Sentinel Window")
    ax4.axhline(0,color="black",linestyle="--")
    ax4.legend(loc="upper right")
    # ax4.set_ylim(0,1e4)
        
    plt.suptitle(f"{distro.capitalize()} Distribution")
    plt.savefig(f"./figures_new_icu/prediction_error_{distro}_fit.png")
    plt.show()

#%%

fig,ax= plt.subplots(1,1,figsize=(15,7.5),sharex=True)
ax.plot(y_full, color="black",label="Real" ,alpha=.8,linestyle="--")

for distro in distros:
    for i,fit_results in enumerate(fit_results_by_window):    
        if fit_results[distro]["minimization_result"].success == False:
            continue
        window = windows[i]
        result = fit_results[distro]
        y = result['curve']
        train_start = window+41-train_width   
        train_end = window
        test_end = train_end + test_width
        
        _y = y[41:train_width] 
        l1, = ax.plot(
            np.arange(train_start,train_end)[:len(_y)],
            _y,
            color=colors[0],
        )
        _y = y[train_width:train_width+test_width]
        l2, = ax.plot(
            np.arange(train_end,test_end)[:len(_y)],
            _y,
            color=colors[1],
        )
    
ax.set_ylim(-100,6000)
ax.grid()

ax.plot([],[],color=colors[0],label = f"All Distros Train")
ax.plot([],[],color=colors[1],label = f"All Distros Prediction")
ax.legend()

plt.savefig(f"./figures_new_icu/prediction_error_all_fits.png")
plt.show()
#%%


# plot distributions

for distro in distros:
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
    plt.show()

#%%
good_distros_en = ["exponential","gaussian","t"]
# find positions, where all of them fit
good_positions = []
bad_positions = []
for i,fit_results in enumerate(fit_results_by_window):
    if all([fit_results[distro]["minimization_result"].success for distro in good_distros_en]):
        good_positions.append(i)
    else:
        bad_positions.append(i)
good_positions = np.array(good_positions)
bad_positions = np.array(bad_positions)

good_train_errors,good_test_errors = [],[]
for distro in good_distros_en:
    train_errors = np.array([fit_results[distro]["train_error"] for fit_results in fit_results_by_window])
    test_errors  = np.array([fit_results[distro]["test_error"] for fit_results in fit_results_by_window])
    good_train_errors.append(train_errors[good_positions])
    good_test_errors.append(test_errors[good_positions])
good_train_errors = np.array(good_train_errors)    
good_test_errors = np.array(good_train_errors)
# means
good_train_errors_mean = np.mean(good_train_errors,axis=1)
good_test_errors_mean = np.mean(good_test_errors,axis=1)
print("Good Positions:",good_positions)
print("Bad Positions:",bad_positions)
print("Good Train Errors:",good_train_errors_mean)
print("Good Test Errors:",good_test_errors_mean)
#%%
# plot means

relative_train_errors = []
relative_test_errors = []
for distro in distros:
    relative_train_errors.append([fit_results[distro]["train_relative_error"] for fit_results in fit_results_by_window])
    relative_test_errors.append([fit_results[distro]["test_relative_error"] for fit_results in fit_results_by_window])
relative_train_errors = np.array(relative_train_errors)
relative_test_errors = np.array(relative_test_errors)
#####################################################################
######### relative errrors ##########################################
#####################################################################


fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,7),sharex=True)
mean_relative_train_errors = np.mean(relative_train_errors,axis=0)
mean_relative_test_errors = np.mean(relative_test_errors,axis=0)

#plot icu and incidence
ax1.plot(df["icu"].values/df["AnzahlFall"].values,label="ICU Incidence Ratio")
ax1.axvspan(sentinel_start_day,sentinel_end_day, color="green", alpha=0.1)


ax2.plot(windows,mean_relative_train_errors,label="Train Error")
ax2.plot(windows,mean_relative_test_errors,label="Test Error")
ax2.axvspan(sentinel_start_day,sentinel_end_day, color="green", alpha=0.1)
ax2.legend()
plt.show()

#%%
fig,ax = plt.subplots(1,1,figsize=(10,5))
transition_rates = np.array([[fit_results[distro]["params"][0] for fit_results in fit_results_by_window] for distro in distros])
for i,distro in enumerate(distros):
    plt.plot(transition_rates[i],label=distro)
plt.legend()
plt.title("Transition Rates")
plt.savefig("./figures_new_icu/transition_rates.png")
plt.show()
# %%
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
    "block",
    "sentinel",
]
for i,distro in enumerate(distros):
    if distro in nono:
        continue
    ax2.plot(windows,transition_rates[i],label=distro)
ax2.axhline(0,color="black",linestyle="--")
ax2.legend()
ax2.set_title("Transition Rates")
# ax2.set_ylim(-.03,0.2)
plt.savefig("./figures_new_icu/transition_rates_and_icu.png")
plt.show()
#%%
# correlate transition rates with error
df_temp = pd.DataFrame()
df_temp["trans_rates"] = transition_rates.reshape(-1)
df_temp["train_errors"] = np.reshape(train_errors_by_distro,-1)
df_temp["test_errors"] = np.reshape(test_errors_by_distro,-1)

sns.pairplot(df_temp)
plt.show()

#%%