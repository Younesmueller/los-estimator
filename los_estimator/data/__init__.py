"""Data loading and preparation utilities for LOS estimation."""

from dataclasses import dataclass
from importlib import resources

import numpy as np
import pandas as pd

from ..config import DataConfig

__all__ = [
    "DataPackage",
    "DataUtils",
    "DataLoader",
]


@dataclass
class DataPackage:
    df_occupancy: pd.DataFrame
    real_los: pd.Series
    df_init: pd.DataFrame
    df_mutant: pd.DataFrame
    xtick_pos: list
    xtick_label: list


class DataUtils:
    def date_to_day(date, start_day):
        """Convert date to day number relative to start_day."""
        return (date - pd.Timestamp(start_day)).days

    def day_to_date(day, start_day):
        """Convert day number to date relative to start_day."""
        return pd.Timestamp(start_day) + pd.Timedelta(days=day)

    def generate_xticks(df):
        xtick_pos = []
        xtick_label = []
        for i in range(0, len(df)):
            if df.index[i].day == 1:  # first day of month
                xtick_pos.append(i)
                label = df.index[i].strftime("%b")
                if i == 0 or df.index[i].month == 1:
                    label += f"\n{df.index[i].year}"
                xtick_label.append(label)
        return xtick_pos, xtick_label


class DataLoader:
    """Data loader for LOS estimation datasets."""

    def __init__(self, data_config: DataConfig):
        self.data_config: DataConfig = data_config
        self.resource_path = str(resources.files("los_estimator").joinpath("data"))

    def _get_packaged_file_path(self, path):
        """Get path to packaged data file using importlib.resources."""
        if "${data}" in path:
            path = path.replace("${data}", self.resource_path)
        return path

    def read_csv(self, path, *args, **kwargs):
        """Read CSV file from packaged data."""
        file_path = self._get_packaged_file_path(path)
        return pd.read_csv(file_path, *args, **kwargs)

    def read_excel(self, path, *args, **kwargs):
        """Read Excel file from packaged data."""
        file_path = self._get_packaged_file_path(path)
        return pd.read_excel(file_path, *args, **kwargs)

    def load_all_data(self):
        c = self.data_config

        real_los: np.array = self._load_los(file=c.los_file)[0]
        df_occupancy: pd.DataFrame = self.load_inc_beds(c.start_day, c.end_day)
        df_init: pd.DataFrame = self.load_init_parameters(c.init_params_file)
        df_mutant: pd.DataFrame = self.load_mutant_distribution(c.mutants_file)
        df_mutant: pd.DataFrame = self.select_mutants(df_occupancy, df_mutant)

        xtick_pos: list = None
        xtick_label: list = None
        xtick_pos, xtick_label = DataUtils.generate_xticks(df_occupancy)

        return DataPackage(
            df_occupancy=df_occupancy,
            real_los=real_los,
            df_init=df_init,
            df_mutant=df_mutant,
            xtick_pos=xtick_pos,
            xtick_label=xtick_label,
        )

    def select_mutants(self, df_occupancy, df_mutant):
        df_mutant = df_mutant.reindex(df_occupancy.index, method="nearest")
        df_mutant["Omikron_BA.1/2"] = df_mutant["Omikron_BA.1"] + df_mutant["Omikron_BA.2"]
        df_mutant["Omikron_BA.4/5"] = df_mutant["Omikron_BA.4"] + df_mutant["Omikron_BA.5"]
        df_mutant = df_mutant[["Delta_AY.1", "Omikron_BA.1/2", "Omikron_BA.4/5"]]
        return df_mutant

    def load_inc_beds(self, start_day, end_day):
        df_inc, _ = self._load_incidences(start_day, end_day)
        df_icu = self._load_icu_occupancy(start_day, end_day)

        df = df_inc.join(df_icu, how="inner")

        df["new_icu_smooth"] = df["new_icu"].rolling(7).mean()
        return df

    def load_init_parameters(self, file):
        df_init = self.read_csv(file, index_col=0)
        df_init = df_init.set_index("distro")
        # interpret model_config as array float of format [f1 f2 f3 ...]
        df_init["params"] = df_init["params"].apply(lambda x: [float(i) for i in x[1:-1].split()])
        return df_init

    def _load_los(self, cutoff_percentage=0.9, file=""):
        df_los = self.read_csv(file, index_col=0)

        los = df_los.iloc[:, 0].to_numpy(dtype=float)
        los /= los.sum()

        los_cutoff = np.argmax(los.cumsum() > cutoff_percentage)
        return los, los_cutoff

    def _load_incidences(self, start_day, end_day):
        """cases_*.csv contains the germany wide covid data provided from RKI. It is derived from https://github.com/robert-koch-institut/Intensivkapazitaeten_und_COVID-19-Intensivbettenbelegung_in_Deutschland/blob/main/Intensivregister_Bundeslaender_Kapazitaeten.csv"""
        c = self.data_config
        df_inc = self.read_csv(c.cases_file, index_col=0, parse_dates=["Refdatum"])
        raw = df_inc.copy()
        df_inc = df_inc[["AnzahlFall", "daily"]]

        date_range = pd.date_range(start="2020-01-02", end=df_inc.index.min(), inclusive="left")
        new_data = pd.DataFrame(0, index=date_range, columns=df_inc.columns)
        df_inc = pd.concat([new_data, df_inc])

        df_inc = df_inc[df_inc.index >= start_day]
        df_inc = df_inc[df_inc.index <= end_day]
        return df_inc, raw

    def _load_icu_occupancy(self, start_day, end_day):
        """icu_occupancy_*.csv contains the germany wide used icu cases.
        It contains the number of occupied beds. From apprx. june on it also contains the number of newly admitted patients.
        It is derived from https://github.com/robert-koch-institut/Intensivkapazitaeten_und_COVID-19-Intensivbettenbelegung_in_Deutschland/blob/main/Intensivregister_Bundeslaender_Kapazitaeten.csv
        """

        c = self.data_config

        df_icu = self.read_csv(c.icu_occupancy_file, parse_dates=["datum"])
        df_icu = df_icu[["datum", "faelle_covid_aktuell", "faelle_covid_erstaufnahmen"]]
        df_icu.columns = ["datum", "icu", "new_icu"]
        df_icu = df_icu.groupby("datum").sum()

        # Fill up the time from beginning of 2020 to data begin
        date_range = pd.date_range(start="2020-01-01", end=df_icu.index.min(), inclusive="left")
        new_data = pd.DataFrame(0, index=date_range, columns=df_icu.columns)
        df_icu = pd.concat([new_data, df_icu])

        df_icu = df_icu[df_icu.index >= start_day]
        df_icu = df_icu[df_icu.index <= end_day]
        return df_icu

    def load_mutant_distribution(self, path):
        raw_df = self.read_excel(path, sheet_name=1)
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
