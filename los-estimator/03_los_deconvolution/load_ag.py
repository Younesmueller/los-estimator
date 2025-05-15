#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#%%
start_day = "2020-01-01"
end_day = "2025-01-01"

df_adm = pd.read_csv("../data/Intensivregister_Bundeslaender_Kapazitaeten.csv",parse_dates=["datum"])
df_adm.set_index("datum", inplace=True)
df_adm = df_adm[["faelle_covid_erstaufnahmen"]]
df_adm = df_adm.groupby("datum").sum()

df_age = pd.read_csv("../data/Intensivregister_Deutschland_Altersgruppen.csv",parse_dates=["datum"])
df_age.set_index("datum", inplace=True)
df_age.drop(columns=["bundesland_id","bundesland_name"], inplace=True)

# aggregate Age groups
cols = ['altersgruppe_0_bis_17','altersgruppe_18_bis_29', 'altersgruppe_30_bis_39','altersgruppe_40_bis_49', 'altersgruppe_50_bis_59']
df_age["altesrgruppe_unter_60"] = df_age[cols].sum(axis=1)
df_age = df_age.drop(columns=cols)
cols = ['altersgruppe_60_bis_69','altersgruppe_70_bis_79']
df_age["altesrgruppe_60_80"] = df_age[cols].sum(axis=1)
df_age = df_age.drop(columns=cols)
# add to other columns
cols = ['altesrgruppe_unter_60','altesrgruppe_60_80','altersgruppe_80_plus']
for col in cols:
    df_age[col] += df_age["altersgruppe_unbekannt"] / len(cols)
df_age = df_age.drop(columns=["altersgruppe_unbekannt"])
date_range = pd.date_range(start="2020-01-01", end=df_age.index.min())
new_data = pd.DataFrame(0, index=date_range, columns=df_age.columns)
df_age = pd.concat([new_data, df_age])
# cut off at start end end date
df_age = df_age[df_age.index >= start_day]
df_age = df_age[df_age.index <= end_day]
figure = plt.figure(figsize=(10,5))
df_age.plot(figsize=(10,5))

#%%
df_age["all"] = df_age[cols].sum(axis=1)
for col in cols:
    df_age[col] /= df_age["all"]
df_age.drop(columns=["all"], inplace=True)
for col in cols:
    df_age[col] *= df_adm["faelle_covid_erstaufnahmen"]

# append zeros at beginning
date_range = pd.date_range(start="2020-01-01", end=df_age.index.min())
new_data = pd.DataFrame(0, index=date_range, columns=df_age.columns)
df_age = pd.concat([new_data, df_age])
# cut off at start end end date
df_age = df_age[df_age.index >= start_day]
df_age = df_age[df_age.index <= end_day]
df_age = df_age[df_age.index>="2021-07-01"]
df_age.plot()
# %%
