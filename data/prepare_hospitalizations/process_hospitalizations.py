#%%
import pandas as pd
import numpy as np
import os
from functools import lru_cache
import matplotlib.pyplot as plt

@lru_cache
def load_case_raw(file):
    return pd.read_csv(file)
#%%
file = "./Aktuell_Deutschland_COVID-19-Hospitalisierungen.csv"
df = load_case_raw(file)
df = df[df["Bundesland_Id"] == 0]
df = df[df["Altersgruppe"] != "00+"]
df = df.drop(columns=["Bundesland","Bundesland_Id","7T_Hospitalisierung_Inzidenz"])
ags = df["Altersgruppe"].unique()
df2 = df.groupby(["Datum","Altersgruppe"]).sum()
df2 = df2.reset_index().set_index("Datum")
fig, axs = plt.subplots(3,2,sharex=True,sharey=True,figsize=(12,5))
for ag,ax in zip(ags,axs.flatten()):
    df2[df2["Altersgruppe"] == ag].plot(ax=ax)
    ax.set_title(ag)
plt.tight_layout()
plt.show()
#%%
# Extract Age profile
df_pivoted = df.pivot(index='Datum', columns='Altersgruppe', values='7T_Hospitalisierung_Faelle')
df_pivoted.plot(subplots=True,sharey=True)

#%%
df_pivoted.to_csv("../hosp_ag.csv",index=True)
# df.to_csv("cases.csv", index=True)
#%%

df["AnzahlFall"].plot()