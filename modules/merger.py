import pandas as pd
import hashlib

from modules.ai_engine import ai_match_columns



def generate_ai_id(df):


    important = df.columns[:3]


    combined = (

        df[important]
        .astype(str)
        .agg("_".join,axis=1)

    )


    df["AI_GENERATED_ID"] = [

        hashlib.md5(
            x.encode()
        )
        .hexdigest()[:10]


        for x in combined
    ]



    return df





def smart_ai_merge(
        df1,
        df2
):


    matches=ai_match_columns(
        df1,
        df2
    )



    print(matches)



    if len(matches)>0:


        best=max(

            matches,

            key=lambda x:x["confidence"]

        )


        merged=pd.merge(

            df1,

            df2,

            left_on=
            best["file1_column"],

            right_on=
            best["file2_column"],

            how="outer"

        )


        return merged,matches



    else:


        df1=generate_ai_id(df1)

        df2=generate_ai_id(df2)



        merged=pd.merge(

            df1,
            df2,

            on="AI_GENERATED_ID",

            how="outer"

        )


        return merged,[]
