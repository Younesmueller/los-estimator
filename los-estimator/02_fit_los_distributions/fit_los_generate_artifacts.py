#%%
%load_ext autoreload
%autoreload 2
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
sys.path.append("../03_los_deconvolution/")
from los_fitter import mse, distributions, generate_kernel, objective_fit_kernel_to_sentinel

#%%
# Inputs
los_distro_csv = "../01_create_los_profiles/berlin/output_los/los_berlin_all.csv"
name = "los_berlin_all"

init_stretch = 1
#%%
folder = f"output_los/{name}"
figures_folder = f"{folder}/figures"

#%%

if not os.path.exists(folder):
    os.makedirs(folder)
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder)

#%%
df_raw = pd.read_csv(los_distro_csv, index_col=0)
los_raw = df_raw['count'].values
los_raw = los_raw.astype(float)
los_raw /= los_raw.sum()  # Normalize the kernel


#%%
# rmove block and sentinel from distributions if present
distros = distributions.copy()
distros.pop("block")
distros.pop("sentinel")

# Example real kernel data
real_kernel = los_raw.copy()

errors = {}
fit_results = {}
result_dict ={}
for distro, init_params in distros.items():
    params = [*init_params, init_stretch]
    result = minimize(
        objective_fit_kernel_to_sentinel(distro),
        params,
        args=(real_kernel,),
        method="L-BFGS-B",
        )
    fit_results[distro] = result.x
    errors[distro] = result.fun
    if not result.success:
        print(f"Failed to fit {distro}")
        print(result.message)
    result_dict[distro] = result


sorted_errors = {k: v for k, v in sorted(errors.items(), key=lambda item: item[1])}
print("Errors:")
for distro, error in sorted_errors.items():
    print(f"{distro.capitalize()}: {error:.6f}")
# Plot errors
plt.bar(sorted_errors.keys(), sorted_errors.values())
plt.ylabel("MSE")
plt.title("Errors")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{figures_folder}/errors.png")
plt.show()

#%%
for distro in fit_results.keys():
    result = result_dict[distro]
    if not result.success:
        continue
    # Generate and plot the fitted kernel
    fitted_kernel = generate_kernel(distro, result.x, real_kernel.shape[0])
    print(f"{distro.capitalize()} params:", result.x)
    print(f"Error ({distro.capitalize()}):", mse(fitted_kernel, real_kernel))
    plt.plot(fitted_kernel, label=f"Estimated {distro.capitalize()}")
    plt.plot(real_kernel, label="Real")
    plt.legend()
    plt.title(distro)
    plt.savefig(f"{figures_folder}/fit_line_{distro}.png")    
    plt.show()
#%%

# Plot all fits together for comparison
for distro, params in fit_results.items():
    fitted_kernel = generate_kernel(distro, params, real_kernel.shape[0])
    plt.plot(fitted_kernel, label=distro.capitalize())

plt.bar(np.arange(len(real_kernel)), real_kernel, alpha=0.5)
plt.ylim(0, 0.06)
plt.legend()
plt.title("Fit Probability Distribution to Sentinel LoS Distribution")
plt.xlim(0,50)
plt.xlabel("days")
plt.ylabel("discharge probability")
plt.savefig(f"{figures_folder}/fit_line_all.png")
plt.show()
#%%
# Print errors in order

#%%
# Plot all fits in their own plots
# Plot all fits together for comparison
for distro, params in fit_results.items():
    fitted_kernel = generate_kernel(distro, params, real_kernel.shape[0])
    plt.bar(range(len(fitted_kernel)),fitted_kernel, label=distro.capitalize())
    plt.plot(real_kernel, label="Real", color='black')
    plt.ylim(0, 0.06)
    plt.xlim(0,50)
    plt.legend()
    plt.title(f"{distro.capitalize()} - Error: {errors[distro]:.6f}")
    plt.savefig(f"{figures_folder}/fit_bar_{distro}.png")
    plt.show()
#%%
fig,axs = plt.subplots(3,3,sharex=True,sharey=True,figsize=(10,10),dpi=200)
axs = axs.flatten()
for i, (distro, params) in enumerate(fit_results.items()):
    ax = axs[i]
    fitted_kernel = generate_kernel(distro, params, real_kernel.shape[0])
    ax.bar(range(len(fitted_kernel)),fitted_kernel, label=distro.capitalize())    
    ax.plot(real_kernel, label="Real", color='black')
    ax.legend()
    ax.set_title(f"{distro.capitalize()} - Error: {errors[distro]:.2e}")
plt.ylim(0, 0.06)
plt.xlim(0,50)
plt.tight_layout() 
plt.savefig(f"{figures_folder}/fit_bar_all.png")
plt.show()
#%%
# Save parameters and errors
l_fit_results = []
for distro, params in fit_results.items():
    error = errors[distro]
    l_fit_results.append((distro,params,error))

fit_results_df = pd.DataFrame(l_fit_results, columns=["distro", "params", "error"])
fit_results_df.to_csv(f"{folder}/fit_results.csv")
#%%


distro = "linear"
params = [40,1.]
x = np.arange(len(real_kernel))
fitted_kernel = -x/params[0]+1
fitted_kernel[fitted_kernel<0] = 0
fitted_kernel /= fitted_kernel.sum()
# fitted_kernel = generate_kernel(distro, params, real_kernel.shape[0])
plt.bar(range(len(fitted_kernel)),fitted_kernel, label=distro.capitalize())    
plt.plot(real_kernel, label="Real", color='black')
plt.legend()
plt.title(f"{distro.capitalize()} - Error: {errors[distro]:.2e}")
plt.show()
#%%
import math
# two part linear

params = [25,.15,75]
# params = [25.5,1.5,75]
p1x, p1y, p2x = params
plt.figure(figsize=(10,5))
p1x_int = math.floor(p1x+1)
p2x_int = math.floor(p2x+1)

x = np.arange(len(real_kernel))
kernel = np.zeros(len(real_kernel))

m = (p1y-1)/p1x
kernel[:p1x_int] = m*x[:p1x_int] + 1

m2 = -p1y/(p2x-p1x)
b2 = p1y - m2*p1x
kernel[p1x_int:p2x_int] = (m2*x + b2)[p1x_int:p2x_int]

kernel /= kernel.sum()
plt.bar(range(len(kernel)),kernel, label=distro.capitalize())    
plt.plot(real_kernel, label="Real", color='black')
plt.grid()
plt.legend()
plt.title(f"{distro.capitalize()} - Error: {errors[distro]:.2e}")
plt.show()
#%%
