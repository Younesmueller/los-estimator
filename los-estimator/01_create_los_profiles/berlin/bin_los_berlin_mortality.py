# %%
import sys
import pandas as pd
import numpy as np

sys.path.append("../")
from binning import perform_binning_and_plot
# %%
df_raw = pd.read_excel("Tabelle_covid_berlin.xlsx")
norm_col = "Verweildauer level  normal"
mv_col = "Verweildauer level MV"
df_raw
#%%
df = df_raw.copy()


def remove_by_criterion(df, criterion, comment=""):
    before = len(df)
    removed = df[~criterion]
    df = df[criterion]
    print(f"Removed {before - len(df)} rows by:", comment)
    return df, removed


criterion = (df["Alter"] >= 0) & (df["Alter"] < 130)
df, removed = remove_by_criterion(df, criterion, "Age")
criterion = df["Aufnahmedatum"] > "2020-01-01"
df, removed = remove_by_criterion(df, criterion, "Aufnahmedatum")
criterion = df["Aufnahmedatum"] <= df["Entlassdatum"]
df, removed = remove_by_criterion(df, criterion, "Aufnahmedatum > Entlassdatum")

# %%
# TODO: Dataprep: Passen Entlassdatum - Aufnahmedatum zusammen mit Verweildauer normal und verweildauaer MV
los = df["Entlassdatum"] - df["Aufnahmedatum"]
df["los"] = los.dt.days
#%%
df["Entlassart"].value_counts()
#%%
df["Entlassart"].value_counts()
df_died = df[df["Entlassart"] == "verstorben"]
df_normal = df[df["Entlassart"] == "Entlassung auf Normalstation"]
#%%
import matplotlib.pyplot as plt
def do_stuff(df):
    df2 = df.copy()
    bins = np.arange(0, df2["los"].max(), 1)
    col = f"los_mv_bin"
    df2[col] = pd.cut(df2["los"], bins,include_lowest=True,right=False)
    df2[col] = df2[col].apply(lambda x: x.left)
    df_binned = df2[col].value_counts().sort_index()
    return df_binned

los_all = do_stuff(df)
los_died = do_stuff(df_died)
los_survived = do_stuff(df_normal)

los_died/=los_died.sum()
los_survived/=los_survived.sum()
los_all/=los_all.sum()
plt.plot(los_died, label="Died")
plt.plot(los_survived, label="Survived")
plt.plot(los_all, label="All")
plt.legend()
plt.show()
#%%
#smooth with a two day window
def smooth(df,days=2):
    df2 = df.copy()
    df2 = df2.rolling(window=days).mean()
    return df2
los_died_smoothed = smooth(los_died)
los_survived_smoothed = smooth(los_survived)
los_all_smoothed = smooth(los_all)
plt.plot(los_died_smoothed, label="Died")
plt.plot(los_survived_smoothed, label="Survived")
# plt.plot(los_all_smoothed, label="All")
plt.legend()
plt.title("LoS Smoothed over 2 days")
plt.show()


#%%

perform_binning_and_plot(df[["los"]], "los_berlin_all")


#%%
# Categories to distinguish between
# 1. Age Group
# 2. Time of pandemic
# 3. Level of CAre
df2 = pd.DataFrame(df[["los", "Altersgruppe"]])
# %% Age Groups
ags = df2["Altersgruppe"].unique()
for ag in ags:
    df_ag = df2[df2["Altersgruppe"] == ag]
    perform_binning_and_plot(df_ag, f"los_berlin_{ag}")

# %% Time of pandemic
df2 = pd.DataFrame(df[["los", "Aufnahmedatum"]])
df2["Aufnahmedatum"] = pd.to_datetime(df2["Aufnahmedatum"])
# bin by month and year
df2["month"] = df2["Aufnahmedatum"].dt.to_period("M")
months = df2["month"].unique()
for month in months:
    df_month = df2[df2["month"] == month]
    perform_binning_and_plot(df_month, f"los_berlin_{month}")



# %%
