# %%
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
import pandas as pd

sys.path.append("../")
from binning import perform_binning_and_plot

# %%

col_ag = "Alterskategorie"

path = "Workdata_AC.xlsx"
df_raw_los = pd.read_excel(path, skiprows=1)
df_raw_los

# %%
# First patient in July - 21.07.2021 - is Nr. 752. The offset between excel and pandas is 3
# The last Patient is admitted on 14.12.2021
# Before 21.07.2021: Alpha, after: Delta
# Decide to Ignore Split point, because there are only 41 Patients in Delta
# july_patients_split_point = 752 - 3 
# df = df_raw_los.iloc[:july_patients_split_point]
itc_col = "Verweildauer\nIntensivstation"
df = df_raw_los.copy()
df = df[df["Intensivpatient"] == "j"]
df["los"] =df[itc_col]
df = df[["los", col_ag]]

#%%
df_binned = perform_binning_and_plot(df, "los_aachen_all")
df_binned.plot()
plt.show()
#%%
remove = [" ","≥","<"]
# remove characters in dataframe
df2 = df.copy()
for r in remove:
    df2[col_ag] = df2[col_ag].str.replace(r, "")
ags = sorted(df2[col_ag].unique())
# clean ag for filename
for ag in ags:
    df3 = df2[df2[col_ag] == ag]
    df_binned = perform_binning_and_plot(df3, f"los_aachen_{ag}")
    df_binned.plot()
    plt.title(f"Age Group: {ag}, n={len(df3)}")
    plt.show()

# %%
