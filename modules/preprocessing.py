import pandas as pd

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.model_selection import train_test_split




def encode(df):


    encoder=LabelEncoder()


    for col in df.columns:


        if df[col].dtype=="object":


            df[col]=encoder.fit_transform(

                df[col].astype(str)

            )


    return df





def scale(df):


    scaler=StandardScaler()


    result=scaler.fit_transform(df)


    return pd.DataFrame(
        result,
        columns=df.columns
    )





def feature_selection(df):


    constant=[]


    for col in df.columns:

        if df[col].nunique()==1:

            constant.append(col)


    df.drop(
        constant,
        axis=1,
        inplace=True
    )


    return df




def ml_ready(df):


    df=encode(df)


    df=feature_selection(df)


    df=scale(df)



    train,test=train_test_split(

        df,

        test_size=.2,

        random_state=42

    )


    return train,test
