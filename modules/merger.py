import pandas as pd
from fuzzywuzzy import fuzz



def find_common_columns(df1,df2):

    common=[]

    for c1 in df1.columns:

        for c2 in df2.columns:


            score=fuzz.ratio(
                c1.lower(),
                c2.lower()
            )


            if score>75:

                common.append(
                    (c1,c2)
                )

    return common





def create_unique_id(df):


    main=df.columns[0]


    df["AUTO_ID"] = (

        main[:3].upper()

        +

        "_"

        +

        df[main]
        .astype(str)
        .str[:5]

        +

        "_"

        +

        df.index.astype(str)

    )


    return df





def merge_files(df1,df2):


    matches=find_common_columns(
        df1,
        df2
    )


    if len(matches)>0:


        left,right=matches[0]


        merged=pd.merge(

            df1,
            df2,

            left_on=left,

            right_on=right,

            how="outer"

        )


        return merged



    else:


        df1=create_unique_id(df1)

        df2=create_unique_id(df2)


        return pd.concat(
            [df1,df2],
            axis=1
        )
