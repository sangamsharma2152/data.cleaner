import pandas as pd
from difflib import SequenceMatcher



# =================================================
# SIMILARITY ENGINE
# =================================================

def similarity(a, b):

    return SequenceMatcher(
        None,
        str(a).lower(),
        str(b).lower()
    ).ratio()



# =================================================
# NORMALIZATION
# =================================================

def clean_value(x):

    x = str(x).lower()


    replacements = [
        "gb",
        "mb",
        "₹",
        "$",
        ","
    ]


    for r in replacements:

        x = x.replace(
            r,
            ""
        )


    return x.strip()




def normalize_dataframe(df):

    df = df.copy()


    df.columns = (

        df.columns
        .str.lower()
        .str.strip()
        .str.replace(
            " ",
            "_"
        )

    )



    for col in df.columns:


        df[col] = (

            df[col]
            .astype(str)
            .apply(clean_value)

        )


    return df




# =================================================
# DATA OVERLAP CHECK
# =================================================

def calculate_overlap(
        s1,
        s2
):


    a = set(

        s1.dropna()
        .astype(str)
        .head(300)

    )



    b = set(

        s2.dropna()
        .astype(str)
        .head(300)

    )



    if len(a)==0:

        return 0



    return (

        len(
            a.intersection(b)
        )

        /

        len(a)

    )




# =================================================
# FIND BEST MATCHES
# =================================================

def find_matches(
        df1,
        df2
):


    results=[]



    for c1 in df1.columns:



        for c2 in df2.columns:



            column_score = similarity(
                c1,
                c2
            )



            value_score = calculate_overlap(

                df1[c1],

                df2[c2]

            )




            final_score = (

                column_score*0.4

                +

                value_score*0.6

            )




            results.append(

                {

                "file1_column":c1,

                "file2_column":c2,

                "column_similarity":round(column_score,2),

                "value_overlap":round(value_score,2),

                "confidence":round(final_score,2)

                }

            )




    return sorted(

        results,

        key=lambda x:x["confidence"],

        reverse=True

    )





# =================================================
# MERGE PREVIEW
# =================================================

def merge_preview(
        df1,
        df2
):


    df1=normalize_dataframe(df1)

    df2=normalize_dataframe(df2)


    matches=find_matches(
        df1,
        df2
    )


    if matches:

        return matches[0],matches


    return None,[]






# =================================================
# MANUAL MERGE
# =================================================

def merge_on_columns(

        df1,
        df2,
        left_col,
        right_col,
        how="inner"

):


    df1=normalize_dataframe(df1)

    df2=normalize_dataframe(df2)



    left_col = (

        left_col.lower()
        .replace(" ","_")

    )



    right_col = (

        right_col.lower()
        .replace(" ","_")

    )



    return pd.merge(

        df1,

        df2,

        left_on=left_col,

        right_on=right_col,

        how=how

    )





# =================================================
# AI MERGE
# =================================================

def smart_ai_merge(

        df1,

        df2,

        threshold=.30,

        how="inner",

        match_mode="fast"

):


    df1=normalize_dataframe(df1)

    df2=normalize_dataframe(df2)



    matches=find_matches(

        df1,

        df2

    )



    if len(matches)==0:


        return (

            pd.DataFrame(),

            [],

            {}

        )



    best=matches[0]



    if best["confidence"] < threshold:


        return (

            pd.DataFrame(),

            matches,

            best

        )





    merged=pd.merge(

        df1,

        df2,


        left_on=best["file1_column"],


        right_on=best["file2_column"],


        how=how

    )




    return (

        merged,

        matches,

        best

    )
