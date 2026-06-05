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

            "missing_fixed": 0,

            "outliers_removed": 0

        }



    # --------------------------
    # CLEAN COLUMN NAMES
    # --------------------------

    def clean_columns(self):


        self.df.columns = (

            self.df.columns
            .astype(str)
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


        before=len(self.df)


        self.df = (

            self.df
            .drop_duplicates()

        )


        removed = before-len(self.df)


        self.report[
            "duplicates_removed"
        ] = removed



        self.steps.append(

            f"Removed {removed} duplicate rows"

        )





    # --------------------------
    # SMART TYPE DETECTION
    # --------------------------

    def convert_possible_numbers(self):


        for col in self.df.columns:


            converted = pd.to_numeric(

                self.df[col],

                errors="ignore"

            )


            self.df[col] = converted



        self.steps.append(

            "Detected numerical columns automatically"

        )






    # --------------------------
    # FIX MISSING
    # --------------------------

    def fix_missing(self):


        total_missing = int(

            self.df
            .isna()
            .sum()
            .sum()

        )



        for col in self.df.columns:



            # numeric columns

            if pd.api.types.is_numeric_dtype(

                self.df[col]

            ):



                value = (

                    self.df[col]
                    .median()

                )


                self.df[col] = (

                    self.df[col]
                    .fillna(value)

                )




            # text columns

            else:


                mode_value = (

                    self.df[col]
                    .mode()

                )


                if len(mode_value)>0:


                    fill = mode_value[0]


                else:


                    fill = "unknown"



                self.df[col] = (

                    self.df[col]
                    .fillna(fill)

                )




        self.report[

            "missing_fixed"

        ] = total_missing



        self.steps.append(

            f"Fixed {total_missing} missing values"

        )





    # --------------------------
    # TEXT CLEANING
    # --------------------------

    def clean_text(self):


        text_cols = (

            self.df
            .select_dtypes(
                include="object"
            )
            .columns

        )



        for col in text_cols:


            self.df[col] = (

                self.df[col]
                .astype(str)
                .str.strip()
                .str.lower()

            )




        self.steps.append(

            "Normalized text formatting"

        )





    # --------------------------
    # OUTLIERS
    # --------------------------

    def remove_outliers(self):


        removed=0


        nums = (

            self.df
            .select_dtypes(
                include=np.number
            )

        )



        for col in nums.columns:


            try:


                q1=self.df[col].quantile(.25)

                q3=self.df[col].quantile(.75)


                iqr=q3-q1


                before=len(self.df)


                self.df=self.df[

                    (

                    self.df[col]>=q1-1.5*iqr

                    )

                    &

                    (

                    self.df[col]<=q3+1.5*iqr

                    )

                ]



                removed += before-len(self.df)



            except:


                pass




        self.report[

            "outliers_removed"

        ]=removed



        self.steps.append(

            f"Removed {removed} outliers"

        )





    # --------------------------
    # MAIN
    # --------------------------

    def clean_all(self):


        self.clean_columns()


        self.convert_possible_numbers()


        self.remove_duplicates()


        self.fix_missing()


        self.clean_text()


        self.remove_outliers()




        self.report[

            "final_rows"

        ]=len(self.df)



        self.report[

            "final_columns"

        ]=len(self.df.columns)




        return self.df





    def get_report(self):

        return self.report




    def explain_steps(self):

        return self.steps
