#%% 
import os
import sys
import numpy as np
import pandas as pd
import types
from pathlib import Path

from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

def date_to_day(date, start_day):\
    return (date - pd.Timestamp(start_day)).days
def day_to_date(day, start_day):
    return pd.Timestamp(start_day) + pd.Timedelta(days=day)


class DataLoader:
    """Data loader for LOS estimation datasets."""

    def __init__(self, data_config):
        self.data_config = data_config
    
    def load_all_data(self):
        return self._load_all_data(**self.data_config.__dict__)
    
    def _load_all_data(self, los_file, init_params_file, mutants_file, start_day, end_day, **kwargs):
        if len(kwargs) > 0:
            print("Warning: Unused parameters passed to load_all_data:", kwargs)
        df_occupancy = self.load_inc_beds(start_day, end_day)
        real_los, _ = self._load_los(file=los_file)
        df_init = self.load_init_parameters(init_params_file)
        df_mutant = self.load_mutant_distribution(mutants_file)
        df_mutant = df_mutant.reindex(df_occupancy.index,method="nearest")
        df_mutant["Omikron_BA.1/2"] = df_mutant["Omikron_BA.1"] + df_mutant["Omikron_BA.2"]
        df_mutant["Omikron_BA.4/5"] = df_mutant["Omikron_BA.4"] + df_mutant["Omikron_BA.5"]
        df_mutant = df_mutant[['Delta_AY.1','Omikron_BA.1/2','Omikron_BA.4/5']]


        xtick_pos, xtick_label = self.generate_xticks(df_occupancy)
        new_icu_date = df_occupancy.index[df_occupancy["new_icu"]>0][0]
        new_icu_day = date_to_day(new_icu_date,start_day)
        return types.SimpleNamespace(
            df_occupancy=df_occupancy,
            real_los=real_los,
            df_init=df_init,
            df_mutant=df_mutant,
            xtick_pos=xtick_pos,
            xtick_label=xtick_label,
            new_icu_date=new_icu_date,
            new_icu_day=new_icu_day
        )
    

    def load_inc_beds(self, start_day, end_day):
        df_inc, raw = self._load_incidences(start_day, end_day)
        # wenn die ersten beiden Yeilen der 1.1.2020 sind entferne eine
        if df_inc.index[0] == pd.Timestamp("2020-01-01") and df_inc.index[1] == pd.Timestamp("2020-01-01"):
            df_inc = df_inc.iloc[1:]

        df_icu = self._load_icu_occupancy(start_day, end_day)
        df = df_inc.join(df_icu,how="inner")
        df["new_icu_smooth"] = df["new_icu"].rolling(7).mean()
        return df


    def generate_xticks(self,df):
        xtick_pos = []
        xtick_label = []
        for i in range(0,len(df)):
            if df.index[i].day == 1: # first day of month
                xtick_pos.append(i)
                label = df.index[i].strftime("%b")
                if i == 0 or df.index[i].month == 1:
                    label += f"\n{df.index[i].year}"
                xtick_label.append(label)
        return xtick_pos,xtick_label


    def load_init_parameters(self,file):
        df_init = pd.read_csv(file,index_col=0)
        df_init = df_init.set_index("distro")
        # interpret params as array float of format [f1 f2 f3 ...]
        df_init["params"] = df_init["params"].apply(lambda x: [float(i) for i in x[1:-1].split()])
        return df_init

    def _load_los(self,cutoff_percentage=.9,file=""):
        df_los = pd.read_csv(file,index_col=0)

        los = df_los.iloc[:,0].to_numpy(dtype=float)
        los /= los.sum()

        los_cutoff = np.argmax(los.cumsum() > cutoff_percentage)
        return los,los_cutoff
        
    def _load_incidences(self,start_day, end_day):
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

    def _load_age_groups(self,start_day,end_day):
        
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

    def _load_icu_occupancy(self,start_day, end_day):
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


        return df
    def load_mutant_distribution(self,p = '../data/VOC_VOI_Tabelle.xlsx'):
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
