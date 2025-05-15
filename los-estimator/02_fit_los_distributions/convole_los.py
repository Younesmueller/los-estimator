#%%
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
sys.path.append("../03_los_deconvolution/")
from dataprep import load_los, load_incidences, load_icu_occupancy
from convolutional_model import calc_its_convolution, objective_function_conv
#%%

los_file = "../01_create_los_profiles/berlin/output_los/los_berlin_all.csv"

initial_guess_prob = 0.018

start_day = "2020-01-01"
end_day = "2025-01-01"

train_start_day = "2020-07-01"
train_end_day = "2020-12-01"

def date_to_day(date):
    return (date - pd.Timestamp(start_day)).days
train_start = date_to_day(pd.Timestamp(train_start_day))
train_end = date_to_day(pd.Timestamp(train_end_day))


#%%
daily_los, los_cutoff = load_los(file = los_file)
plt.plot(daily_los,label="LoS Distribution")
plt.axvline(los_cutoff,color="black",label="90% Cutoff")
plt.legend()
plt.title("Length of Stay Distribution")
plt.show()


#%%
# Use fitted kernel instead of generated kernel


# from los_fitter import generate_kernel
# distro = 'invgauss'
# distro_params = [ 1.89166771, -0.11267099 , 0.0750936 ]
# kernel_size = 120
# kernel = generate_kernel(distro,distro_params,kernel_size)

# plt.plot(kernel,label="LoS Distribution")
# plt.axvline(los_cutoff,color="black",label="90% Cutoff")
# plt.legend()
# plt.title("Length of Stay Distribution")
# plt.show()

# daily_los = kernel

#%%
df_inc = load_incidences(start_day, end_day)
df_icu = load_icu_occupancy(start_day, end_day)
df = df_inc.join(df_icu,how="inner")

df.plot(subplots=True)
plt.suptitle("Incidences and ICU Occupancy")
plt.show()


#%%


x_train = df["AnzahlFall"][train_start:train_end]
y_train= df["icu"][train_start:train_end].values


plt.subplots(1,1,figsize=(10,5))
# methods = ["L-BFGS-B","Powell","Nelder-Mead"]
method = "Nelder-Mead"

best = np.inf
for delay in range(0,10):
    result = minimize(
        objective_function_conv,
        [initial_guess_prob,delay],
        args=(x_train,y_train,daily_los,los_cutoff),
        method=method,
        bounds=[(0, 1),(delay,delay)]
    )
    if result.fun < best:
        best = result.fun
        best_result = result
result = best_result

print(method)
print(result)
print(f"Optimal Transition Rate: {result.x[0]}")
print(f"Optimal Delay: {result.x[1]}")
print("------------------------------------------------")
print()
error = result.fun
print(f"Train MSE: {error}")
test_rmse  = objective_function_conv(result.x,df["AnzahlFall"],df["icu"],daily_los,los_cutoff)
print(f"Train & Test RMSE: {test_rmse}")



res = calc_its_convolution(df["AnzahlFall"],daily_los,*result.x, los_cutoff=los_cutoff)
df_res = pd.DataFrame(index=df.index[:len(res)], data=res[:len(df.index)], columns=["Convolutional Model - {method}"])
plt.plot(df_res,label="Convolutional Model")
plt.axvline(df.index[train_start],color="black",linestyle="--",label="Training Window")
plt.axvline(df.index[train_end],color="black",linestyle="--")
plt.plot(df["icu"],label="Real ICU",color="black")
plt.legend(loc="lower right")
plt.title(f"Convolutional Model, Train: {train_start}, {train_end}")
# plt.ylim(0,1000)
plt.ylim(-500,7000)
plt.show()
#%%
