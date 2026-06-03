import pandas as pd
import numpy as np


class DataCleaner:


    def __init__(self,df):

        self.df=df



    def remove_empty(self):

        self.df.dropna(
            how="all",
            inplace=True
        )

        return self.df



    def missing_values(self):

        for col in self.df.columns:

            if self.df[col].dtype=="object":

                self.df[col].fillna(
                    "Unknown",
                    inplace=True
                )

            else:

                self.df[col].fillna(
                    self.df[col].mean(),
                    inplace=True
                )


        return self.df



    def remove_duplicates(self):

        self.df.drop_duplicates(
            inplace=True
        )

        return self.df



    def remove_outliers(self):

        numeric = self.df.select_dtypes(
            include=np.number
        )

        for col in numeric.columns:


            Q1=self.df[col].quantile(.25)

            Q3=self.df[col].quantile(.75)


            IQR=Q3-Q1


            self.df=self.df[

                (self.df[col]>=Q1-1.5*IQR)

                &

                (self.df[col]<=Q3+1.5*IQR)

            ]


        return self.df



    def clean_all(self):

        self.remove_empty()

        self.missing_values()

        self.remove_duplicates()

        self.remove_outliers()

        return self.df
