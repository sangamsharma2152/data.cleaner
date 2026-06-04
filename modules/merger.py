import pandas as pd
from difflib import SequenceMatcher



def similarity(a,b):

    return SequenceMatcher(
        None,
        str(a).lower(),
        str(b).lower()
    ).ratio()



def clean_value(x):

    return (

        str(x)
        .lower()
        .replace("gb","")
        .replace("₹","")
        .replace(",","")
        .strip()

    )




def normalize_dataframe(df):

    df=df.copy()


    df.columns=(

        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ","_")

    )


    for col in df.columns:

        df[col]=df[col].apply(clean_value)


    return df




def find_best_column_match(df1,df2):


    results=[]


    for c1 in df1.columns:


        for c2 in df2.columns:


            column_score = similarity(
                c1,
                c2
            )


            value_score=0


            sample1=df1[c1].dropna().head(30)

            sample2=df2[c2].dropna().head(30)


            matches=[]


            for a in sample1:

                for b in sample2:


                    matches.append(

                        similarity(a,b)

                    )


            if matches:

                value_score=max(matches)



            final_score=(

                column_score*0.4

                +

                value_score*0.6

            )



            results.append({

                "file1_column":c1,

                "file2_column":c2,

                "confidence":round(final_score,2)

            })



    return sorted(

        results,

        key=lambda x:x["confidence"],

        reverse=True

    )





def merge_on_columns(

        df1,
        df2,
        left_col,
        right_col,
        how="outer"
):


    df1=normalize_dataframe(df1)

    df2=normalize_dataframe(df2)



    return pd.merge(

        df1,

        df2,


        left_on=left_col.lower(),

        right_on=right_col.lower(),


        how=how

    )





def smart_ai_merge(

        df1,

        df2,

        threshold=.45,

        how="outer",

        match_mode="fast"

):


    df1=normalize_dataframe(df1)

    df2=normalize_dataframe(df2)


    matches=find_best_column_match(

        df1,

        df2

    )



    best=matches[0]



    if best["confidence"]>=threshold:


        merged=pd.merge(

            df1,

            df2,


            left_on=best["file1_column"],

            right_on=best["file2_column"],


            how=how

        )


        return merged,matches,best



    else:


        raise Exception(

            "No common entity found. Please select columns manually."

        )
