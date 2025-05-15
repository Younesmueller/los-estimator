#%%
import numpy as np
from scipy.optimize import minimize
from scipy.stats import lognorm, weibull_min, norm, expon, gamma, beta, cauchy, t, invgauss
from convolutional_model import calc_its_convolution
import matplotlib.pyplot as plt
#%%


prob,delay,loc,scale, scaling_fac = [0.01854664, 2.        , 0.28104439, 1.78399418, 0.12894231]

n = 30
#%%
x1 = np.arange(n)
kernel1 = cauchy.pdf(x1, loc=loc, scale=scale)
plt.bar(x1, kernel1)
plt.show()
#%%
x2 = np.arange(n) * scaling_fac
kernel2 = cauchy.pdf(x2, loc=loc, scale=scale)
plt.bar(x2, kernel2)
plt.show()
#%%
x3 = np.arange(n) * scaling_fac
kernel3 = cauchy.pdf(x3, loc=loc, scale=scale)
kernel3 /= kernel3.sum()
plt.bar(x3, kernel3)
plt.show()
#%%
fig,axs = plt.subplots(3,1,figsize=(10,8),sharex=True,sharey=True)
xx = np.arange(len(kernel1))
axs[0].bar(xx, kernel1,width=0.5)   
axs[0].plot(kernel1)
axs[1].bar(xx, kernel2,width=0.5)   
axs[2].bar(xx, kernel3,width=0.5)

plt.suptitle('Steps of Kernel Generation - Cauchy distribution',fontsize=16)
axs[0].set_title('1. Generate Kernel from Parameters')
axs[1].set_title('2. Apply a horizontal stretching factor')
axs[2].set_title('3. Normalize the kernel')
plt.tight_layout()
plt.show()

# %%
