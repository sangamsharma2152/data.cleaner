import numpy as np
import pandas as pd


class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.report = {
            "initial_rows": len(self.df),
            "initial_columns": len(self.df.columns),
            "empty_rows_removed": 0,
            "missing_values_filled": 0,
            "duplicates_removed": 0,
            "outliers_removed": 0,
            "final_rows": len(self.df),
            "final_columns": len(self.df.columns),
        }

    def remove_empty(self):
        before = len(self.df)
        self.df = self.df.dropna(how="all")
        self.report["empty_rows_removed"] = before - len(self.df)
        return self.df

    def missing_values(self):
        missing_before = int(self.df.isna().sum().sum())

        for col in self.df.columns:
            if self.df[col].isna().sum() == 0:
                continue

            if pd.api.types.is_numeric_dtype(self.df[col]):
                fill_value = self.df[col].median()
                if pd.isna(fill_value):
                    fill_value = 0
            else:
                mode = self.df[col].dropna().mode()
                fill_value = mode.iloc[0] if not mode.empty else "Unknown"

            self.df[col] = self.df[col].fillna(fill_value)

        self.report["missing_values_filled"] = missing_before - int(
            self.df.isna().sum().sum()
        )
        return self.df

    def remove_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        self.report["duplicates_removed"] = before - len(self.df)
        return self.df

    def remove_outliers(self):
        before = len(self.df)
        numeric_columns = self.df.select_dtypes(include=np.number).columns

        for col in numeric_columns:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1

            if pd.isna(iqr) or iqr == 0:
                continue

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]

        self.report["outliers_removed"] = before - len(self.df)
        return self.df

    def clean_all(self):
        self.remove_empty()
        self.missing_values()
        self.remove_duplicates()
        self.remove_outliers()
        self.report["final_rows"] = len(self.df)
        self.report["final_columns"] = len(self.df.columns)
        return self.df.copy()

    def get_report(self):
        return self.report.copy()
