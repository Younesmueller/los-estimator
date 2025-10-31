import os

import pandas as pd

base_folder = os.path.dirname(__file__)
input_folder = os.path.join(base_folder, "input")
output_folder = os.path.join(base_folder, "output")


def preprocess_icu_occupancy():
    """icu_occupancy_*.csv contains the germany wide used icu cases.
    It contains the number of occupied beds. From apprx. june on it also contains the number of newly admitted patients.
    It is derived from https://github.com/robert-koch-institut/Intensivkapazitaeten_und_COVID-19-Intensivbettenbelegung_in_Deutschland/blob/main/Intensivregister_Bundeslaender_Kapazitaeten.csv
    """

    input_file = os.path.join(input_folder, "Intensivregister_Bundeslaender_Kapazitaeten.csv")

    df_icu = pd.read_csv(input_file, parse_dates=["datum"])
    df_icu = df_icu[["datum", "faelle_covid_aktuell", "faelle_covid_erstaufnahmen"]]
    df_icu.columns = ["datum", "icu_occupancy", "icu_admissions"]
    df_icu = df_icu.groupby("datum").sum()

    # Fill up the time from beginning of 2020 to data begin
    date_range = pd.date_range(start="2020-01-01", end=df_icu.index.min(), inclusive="left")
    new_data = pd.DataFrame(0, index=date_range, columns=df_icu.columns)
    df_icu = pd.concat([new_data, df_icu])

    df_icu["icu_admissions_smooth"] = df_icu["icu_admissions"].rolling(7).mean()

    os.makedirs(output_folder, exist_ok=True)
    df_icu.to_csv(os.path.join(output_folder, "icu.csv"))

    return df_icu


def preprocess_mutant_distributions() -> pd.DataFrame:
    """Load the mutant distribution data."""
    path = os.path.join(input_folder, "VOC_VOI_Tabelle.xlsx")

    raw_df = pd.read_excel(path, sheet_name=1)
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

    df["Omikron_BA.1/2"] = df["Omikron_BA.1"] + df["Omikron_BA.2"]
    df["Omikron_BA.4/5"] = df["Omikron_BA.4"] + df["Omikron_BA.5"]
    df = df[["Delta_AY.1", "Omikron_BA.1/2", "Omikron_BA.4/5"]]

    # Ensure the output subfolder exists
    os.makedirs(output_folder, exist_ok=True)
    df.to_csv(os.path.join(output_folder, "mutant_distribution.csv"))


if __name__ == "__main__":
    print("Preprocessing ICU occupancy data...")
    preprocess_icu_occupancy()
    print("Preprocessing mutant distributions...")
    preprocess_mutant_distributions()
    print("Preprocessing done.")
