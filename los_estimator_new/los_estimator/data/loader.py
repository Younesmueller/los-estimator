"""
Data loading utilities for LOS estimation.
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, Dict, Any
from pathlib import Path


class DataLoader:
    """Data loader for LOS estimation datasets."""
    
    def __init__(self, data_dir: str = None):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Base directory for data files. If None, uses relative paths.
        """
        self.data_dir = Path(data_dir) if data_dir else None
    
    def _resolve_path(self, file_path: str) -> Path:
        """Resolve file path relative to data directory or as absolute path."""
        path = Path(file_path)
        if self.data_dir and not path.is_absolute():
            return self.data_dir / path
        return path
    
    def load_los_distribution(self, file_path: str, cutoff_percentage: float = 0.9) -> Tuple[np.ndarray, int]:
        """
        Load length of stay distribution from CSV file.
        
        Args:
            file_path: Path to LOS distribution CSV file
            cutoff_percentage: Percentage cutoff for LOS distribution
            
        Returns:
            Tuple of (los_distribution, los_cutoff)
        """
        path = self._resolve_path(file_path)
        df_los = pd.read_csv(path, index_col=0)
        
        los = df_los.iloc[:, 0].to_numpy(dtype=float)
        los /= los.sum()  # Normalize to sum to 1
        
        los_cutoff = np.argmax(los.cumsum() > cutoff_percentage)
        return los, los_cutoff
    
    def load_incidences(self, file_path: str, start_day: str, end_day: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load COVID-19 incidence data.
        
        Args:
            file_path: Path to cases CSV file
            start_day: Start date string
            end_day: End date string
            
        Returns:
            Tuple of (processed_dataframe, raw_dataframe)
        """
        path = self._resolve_path(file_path)
        df_inc = pd.read_csv(path, index_col=0, parse_dates=["Refdatum"])
        raw = df_inc.copy()
        
        df_inc = df_inc[["AnzahlFall", "daily"]]
        
        # Pad with zeros from 2020-01-01 to first data point
        date_range = pd.date_range(start="2020-01-01", end=df_inc.index.min(), inclusive='left')
        new_data = pd.DataFrame(0, index=date_range, columns=df_inc.columns)
        df_inc = pd.concat([new_data, df_inc])
        
        # Filter by date range
        df_inc = df_inc[df_inc.index >= start_day]
        df_inc = df_inc[df_inc.index <= end_day]
        
        return df_inc, raw
    
    def load_icu_occupancy(self, file_path: str, start_day: str, end_day: str) -> pd.DataFrame:
        """
        Load ICU occupancy data.
        
        Args:
            file_path: Path to ICU capacity CSV file
            start_day: Start date string
            end_day: End date string
            
        Returns:
            DataFrame with ICU occupancy data
        """
        path = self._resolve_path(file_path)
        df_icu = pd.read_csv(path, parse_dates=["datum"])
        
        # Select and rename columns to match original format
        df_icu = df_icu[["datum", "faelle_covid_aktuell", "faelle_covid_erstaufnahmen"]]
        df_icu.columns = ["datum", "icu", "new_icu"]
        df_icu = df_icu.groupby("datum").sum()
        
        # Fill up the time from beginning of 2020 to data begin
        date_range = pd.date_range(start="2020-01-01", end=df_icu.index.min(), inclusive='left')
        new_data = pd.DataFrame(0, index=date_range, columns=df_icu.columns)
        df_icu = pd.concat([new_data, df_icu])
        
        # Filter by date range
        df_icu = df_icu[df_icu.index >= start_day]
        df_icu = df_icu[df_icu.index <= end_day]
        
        return df_icu
    
    def load_mutant_distribution(self, file_path: str) -> pd.DataFrame:
        """
        Load variant/mutant distribution data from Excel file.
        
        Args:
            file_path: Path to Excel file with variant data
            
        Returns:
            DataFrame with variant distributions over time
        """
        path = self._resolve_path(file_path)
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
        
        return df
    
    def load_initial_parameters(self, file_path: str) -> pd.DataFrame:
        """
        Load initial fitting parameters from CSV file.
        
        Args:
            file_path: Path to initial parameters CSV file
            
        Returns:
            DataFrame with initial parameters for each distribution
        """
        path = self._resolve_path(file_path)
        df_init = pd.read_csv(path, index_col=0)
        df_init = df_init.set_index("distro")
        
        # Parse params column as list of floats
        df_init["params"] = df_init["params"].apply(
            lambda x: [float(i) for i in x[1:-1].split()] if isinstance(x, str) else x
        )
        
        return df_init
    
    def load_complete_dataset(self, los_file: str, incidence_file: str, icu_file: str,
                             init_params_file: str, mutants_file: str,
                             start_day: str, end_day: str) -> Dict[str, Any]:
        """
        Load complete dataset for LOS estimation.
        
        Args:
            los_file: Path to LOS distribution file
            incidence_file: Path to incidence data file
            icu_file: Path to ICU occupancy file
            init_params_file: Path to initial parameters file
            mutants_file: Path to variant distribution file
            start_day: Start date string
            end_day: End date string
            
        Returns:
            Dictionary containing all loaded data
        """
        # Load individual datasets
        real_los, los_cutoff = self.load_los_distribution(los_file)
        df_inc, raw_inc = self.load_incidences(incidence_file, start_day, end_day)
        df_icu = self.load_icu_occupancy(icu_file, start_day, end_day)
        df_init = self.load_initial_parameters(init_params_file)
        df_mutant = self.load_mutant_distribution(mutants_file)
        
        # Combine incidence and ICU data
        df_occupancy = df_inc.join(df_icu, how="inner")
        df_occupancy["new_icu_smooth"] = df_occupancy["new_icu"].rolling(7).mean()
        
        # Align mutant data with occupancy data
        df_mutant = df_mutant.reindex(df_occupancy.index, method="nearest")
        
        # Generate x-axis ticks for plotting
        xtick_pos, xtick_label = self._generate_xticks(df_occupancy)
        
        # Find first day with ICU admissions
        new_icu_date = df_occupancy.index[df_occupancy["new_icu"] > 0][0]
        
        return {
            "df_occupancy": df_occupancy,
            "real_los": real_los,
            "los_cutoff": los_cutoff,
            "df_init": df_init,
            "df_mutant": df_mutant,
            "xtick_pos": xtick_pos,
            "xtick_label": xtick_label,
            "new_icu_date": new_icu_date,
            "raw_incidence": raw_inc
        }
    
    def _generate_xticks(self, df: pd.DataFrame) -> Tuple[list, list]:
        """Generate x-axis tick positions and labels for time series plots."""
        xtick_pos = []
        xtick_label = []
        
        for i in range(len(df)):
            if df.index[i].day == 1:  # First day of month
                xtick_pos.append(i)
                label = df.index[i].strftime("%b")
                if i == 0 or df.index[i].month == 1:
                    label += f"\n{df.index[i].year}"
                xtick_label.append(label)
        
        return xtick_pos, xtick_label
    
    @staticmethod
    def get_manual_transition_rates(df_occupancy: pd.DataFrame) -> np.ndarray:
        """
        Calculate manual transition rates from occupancy data.
        
        Args:
            df_occupancy: DataFrame with occupancy data
            
        Returns:
            Array of transition rates
        """
        # Simple estimation based on changes in occupancy
        transition_rates = np.diff(df_occupancy["icu"].values) / df_occupancy["icu"].values[:-1]
        transition_rates = np.clip(transition_rates, -1, 1)  # Reasonable bounds
        
        # Pad to match original length
        transition_rates = np.append(transition_rates, transition_rates[-1])
        
        return transition_rates
