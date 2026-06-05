import pandas as pd
import numpy as np


class DataCleaner:


    def __init__(self, df):

        self.df = df.copy()

        self.steps = []

        self.report = {

            "initial_rows": len(df),

            "initial_columns": len(df.columns),

            "duplicates_removed": 0,

            "missing_values_fixed": 0,

            "outliers_removed": 0

        }



    # --------------------------
    # COLUMN CLEANING
    # --------------------------

    def clean_column_names(self):

        self.df.columns = (

            self.df.columns
            .str.lower()
            .str.strip()
            .str.replace(" ","_")

        )


        self.steps.append(
            "Standardized column names"
        )



    # --------------------------
    # DUPLICATES
    # --------------------------

    def remove_duplicates(self):

        before = len(self.df)


        self.df.drop_duplicates(
            inplace=True
        )


        removed = before - len(self.df)


        self.report[
            "duplicates_removed"
        ] = removed


        self.steps.append(

            f"Removed {removed} duplicate rows"

        )



    # --------------------------
    # MISSING VALUES
    # --------------------------

    def fix_missing(self):

        missing_before = (

            self.df
            .isna()
            .sum()
            .sum()

        )


        for col in self.df.columns:


            if self.df[col].dtype == "object":


                self.df[col] = (

                    self.df[col]
                    .fillna("unknown")

                )


            else:


                self.df[col] = (

                    self.df[col]
                    .fillna(
                        self.df[col].median()
                    )

                )



        self.report[
            "missing_values_fixed"
        ] = int(missing_before)



        self.steps.append(

            f"Fixed {missing_before} missing values"

        )



    # --------------------------
    # OUTLIERS
    # --------------------------

    def remove_outliers(self):


        numeric = (

            self.df
            .select_dtypes(
                include=np.number
            )

        )


        removed = 0



        for col in numeric.columns:


            q1 = self.df[col].quantile(.25)

            q3 = self.df[col].quantile(.75)


            iqr = q3-q1


            before=len(self.df)



            self.df = self.df[

                (

                    self.df[col]

                    >= q1-1.5*iqr

                )

                &

                (

                    self.df[col]

                    <= q3+1.5*iqr

                )

            ]



            removed += before-len(self.df)



        self.report[
            "outliers_removed"
        ] = removed



        self.steps.append(

            f"Removed {removed} outliers"

        )




    # --------------------------
    # TEXT NORMALIZATION
    # --------------------------

    def clean_text(self):


        for col in self.df.select_dtypes(

            include="object"

        ):


            self.df[col] = (

                self.df[col]
                .astype(str)
                .str.strip()
                .str.lower()

            )


        self.steps.append(

            "Normalized text values"

        )



    # --------------------------
    # RUN EVERYTHING
    # --------------------------

    def clean_all(self):


        self.clean_column_names()


        self.remove_duplicates()


        self.fix_missing()


        self.clean_text()


        self.remove_outliers()



        self.report[

            "final_rows"

        ] = len(self.df)



        self.report[

            "final_columns"

        ] = len(self.df.columns)



        return self.df




    def get_report(self):

        return self.report



    def explain_steps(self):

        return self.steps
