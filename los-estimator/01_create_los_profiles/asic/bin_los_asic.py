#%%
import pandas as pd
import sys

sys.path.append("../")
from binning import perform_binning_and_plot
#%%
df_raw = pd.read_csv("xlos_all.csv")
#%%
##################################################################################
############### Calculate LoS distribution from ASIC Data ########################
##################################################################################
first_date = pd.to_datetime("2019-03-01")

df = df_raw.copy()
df = df[df["1/2 non/covid"]==2]



df["Date"] = pd.to_datetime(first_date) + pd.to_timedelta(df["admission day"], unit="D")
df = df.drop(columns=["1/2 non/covid","admission day"])
df.columns = ["los","admission"]
# as int
df["los"] = df["los"].round().astype(int)
df2 = df[["los"]]
df2
#%%
perform_binning_and_plot(df2, "los_asic_all")

#%%
df2 = pd.DataFrame(df[["los", "admission"]])
df2["admission"] = pd.to_datetime(df2["admission"])
# bin by month and year
df2["month"] = df2["admission"].dt.to_period("M")
months = df2["month"].unique()
for month in months:
    df_month = df2[df2["month"] == month]
    perform_binning_and_plot(df_month, f"los_asic_{month}")


# %%
