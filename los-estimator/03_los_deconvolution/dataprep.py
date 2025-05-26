import pandas as pd
import numpy as np
import os

def load_los(cutoff_percentage=.9,file=""):
    df_los = pd.read_csv(file,index_col=0)

    los = df_los.iloc[:,0].to_numpy(dtype=float)
    los /= los.sum()

    los_cutoff = np.argmax(los.cumsum() > cutoff_percentage)
    return los,los_cutoff
    
def load_incidences(start_day, end_day):
    """cases_*.csv contains the germany wide covid data provided from RKI. It is derived from https://github.com/robert-koch-institut/Intensivkapazitaeten_und_COVID-19-Intensivbettenbelegung_in_Deutschland/blob/main/Intensivregister_Bundeslaender_Kapazitaeten.csv"""
    file = "../data/cases.csv"
    df_inc = pd.read_csv(file,index_col=0,parse_dates=["Refdatum"])
    raw = df_inc.copy()
    df_inc = df_inc[["AnzahlFall","daily"]]

    date_range = pd.date_range(start="2020-01-01", end=df_inc.index.min(),inclusive='left')
    new_data = pd.DataFrame(0, index=date_range, columns=df_inc.columns)
    df_inc = pd.concat([new_data,df_inc])
    
    df_inc = df_inc[df_inc.index >= start_day]
    df_inc = df_inc[df_inc.index <= end_day]
    return df_inc,raw

def load_age_groups(start_day,end_day):
    
    df_age = pd.read_csv("../data/Intensivregister_Deutschland_Altersgruppen.csv",parse_dates=["datum"])
    df_age.set_index("datum", inplace=True)
    df_age.drop(columns=["bundesland_id","bundesland_name"], inplace=True)

    # # aggregate Age groups
    # cols = ['altersgruppe_0_bis_17','altersgruppe_18_bis_29', 'altersgruppe_30_bis_39','altersgruppe_40_bis_49', 'altersgruppe_50_bis_59']
    # df_age["altesrgruppe_unter_60"] = df_age[cols].sum(axis=1)
    # df_age = df_age.drop(columns=cols)
    # cols = ['altersgruppe_60_bis_69','altersgruppe_70_bis_79']
    # df_age["altesrgruppe_60_80"] = df_age[cols].sum(axis=1)
    # df_age = df_age.drop(columns=cols)
    # # add to other columns
    # cols = ['altesrgruppe_unter_60','altesrgruppe_60_80','altersgruppe_80_plus']
    # for col in cols:
    #     df_age[col] += df_age["altersgruppe_unbekannt"] / len(cols)
    # df_age = df_age.drop(columns=["altersgruppe_unbekannt"])
    date_range = pd.date_range(start="2020-01-01", end=df_age.index.min())
    new_data = pd.DataFrame(0, index=date_range, columns=df_age.columns)
    df_age = pd.concat([new_data, df_age])
    # cut off at start end end date
    df_age = df_age[df_age.index >= start_day]
    df_age = df_age[df_age.index <= end_day]
    return df_age

def load_icu_occupancy(start_day, end_day):
    """icu_occupancy_*.csv contains the germany wide used icu cases.
    It contains the number of occupied beds. From apprx. june on it also contains the number of newly admitted patients.
    It is derived from https://github.com/robert-koch-institut/Intensivkapazitaeten_und_COVID-19-Intensivbettenbelegung_in_Deutschland/blob/main/Intensivregister_Bundeslaender_Kapazitaeten.csv"""


    df_icu = pd.read_csv("../data/Intensivregister_Bundeslaender_Kapazitaeten.csv",parse_dates=["datum"])
    df_icu = df_icu[["datum", "faelle_covid_aktuell","faelle_covid_erstaufnahmen"]]
    df_icu.columns = ["datum","icu","new_icu"]
    df_icu = df_icu.groupby("datum").sum()

    # Fill up the time from beginning of 2020 to data begin
    date_range = pd.date_range(start="2020-01-01", end=df_icu.index.min(),inclusive='left')
    new_data = pd.DataFrame(0, index=date_range, columns=df_icu.columns)
    df_icu = pd.concat([new_data, df_icu])

    df_icu = df_icu[df_icu.index >= start_day]
    df_icu = df_icu[df_icu.index <= end_day]
    return df_icu


def load_mutant_distribution(p = '../data/VOC_VOI_Tabelle.xlsx'):
    raw_df = pd.read_excel(p, sheet_name=1)
    c = [col for col in raw_df.columns if "Anteil" in col]
    df = raw_df[c]
    c = [col.split("+")[0] for col in df.columns]
    df.columns = c
    df = df.iloc[:-1, :]
    KW1 = "2021-01-04"
    df["date"] = pd.date_range(start=KW1, periods=len(df), freq="7D")
    df = df.set_index("date")
    one_week_after = df.index[-1] + pd.Timedelta(days=7)
    dr = pd.date_range(start=df.index[0], end=one_week_after, freq="D")
    df = df.reindex(dr, method="ffill")
    dr = pd.date_range(start="2020-03-10", end=one_week_after, freq="D")
    df = df.reindex(dr)
    df = df.fillna(0)
    df["wildtype"] = 100 - df.sum(axis=1)
    df.loc["2022-05-05":, "wildtype"] = 0

    df = df[[df.columns[-1]] + list(df.columns[:-1])]

    return df