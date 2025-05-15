# %%
import sys
import pandas as pd

sys.path.append("../")
from binning import perform_binning_and_plot
import matplotlib.pyplot as plt
# %%
df_raw = pd.read_excel("Tabelle_covid_berlin.xlsx")
norm_col = "Verweildauer level  normal"
mv_col = "Verweildauer level MV"
df_raw
# %%
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

perform_binning_and_plot(df[["los"]], "los_berlin_all")


#%%
# Categories to distinguish between
# 1. Age Group
# 2. Time of pandemic
# 3. Level of CAre
df2 = pd.DataFrame(df[["los", "Altersgruppe"]])
# %% Age Groups
ags = df2["Altersgruppe"].unique()
ags = sorted(ags)
dfs = []
for ag in ags:
    df_ag = df2[df2["Altersgruppe"] == ag]
    df_ = perform_binning_and_plot(df_ag, f"los_berlin_{ag}")
    dfs.append((df_,len(df_ag)))
#%%
fig,axs = plt.subplots(1,5,sharex=True,sharey=True,figsize=(10,3))
axs = axs.flatten()
for i,(df_,length) in enumerate(dfs):
    ax = axs[i]
    df_.plot(ax=ax)
    ax.set_title(f"{ags[i]} - n={length}")

ax.set_xlim(0,100)
plt.suptitle("LOS Berlin by Age Group", fontsize=16)
plt.tight_layout()
plt.savefig("./figures/los_berlin_by_age_group.png")
plt.plot()

# %% Time of pandemic
df2 = pd.DataFrame(df[["los", "Aufnahmedatum"]])
df2["Aufnahmedatum"] = pd.to_datetime(df2["Aufnahmedatum"])
# bin by month and year
df2["month"] = df2["Aufnahmedatum"].dt.to_period("M")
months = df2["month"].unique()
# sort
months = sorted(months)
dfs = []
for month in months:
    df_month = df2[df2["month"] == month]
    df_ = perform_binning_and_plot(df_month, f"los_berlin_{month}")
    dfs.append((df_,len(df_month)))
#%%
fig,axs = plt.subplots(3,4,sharex=True,sharey=True,figsize=(10,5),dpi=300)
axs = axs.flatten()
for i,(df_,length) in enumerate(dfs):
    ax = axs[i]
    df_.plot(ax=ax)
    ax.set_title(f"{months[i]} - n={length}")

ax.set_xlim(0,100)
plt.suptitle("LOS Berlin by Month", fontsize=16)
plt.tight_layout()
plt.savefig("./figures/los_berlin_by_month.png")
plt.plot()
#%%
