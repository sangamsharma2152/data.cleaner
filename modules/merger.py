import pandas as pd
from difflib import SequenceMatcher



# ==================================================
# TEXT SIMILARITY
# ==================================================

def similarity(a,b):

    return SequenceMatcher(
        None,
        str(a).lower(),
        str(b).lower()
    ).ratio()



# ==================================================
# SAFE COLUMN NORMALIZATION ONLY
# ==================================================

def normalize_columns(df):

    df=df.copy()

    df.columns=(

        df.columns
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(" ","_")

    )

    return df




# ==================================================
# VALUE OVERLAP CHECK
# ==================================================

def value_overlap(a,b):

    try:

        a=set(
            a.dropna()
            .astype(str)
            .str.lower()
            .head(500)
        )

        b=set(
            b.dropna()
            .astype(str)
            .str.lower()
            .head(500)
        )


        if len(a)==0:

            return 0


        return len(a & b)/len(a)


    except:

        return 0






# ==================================================
# FIND COLUMN MATCHES
# ==================================================

def find_matches(df1,df2):


    results=[]


    for c1 in df1.columns:


        for c2 in df2.columns:



            name_score = similarity(
                c1,
                c2
            )


            data_score=value_overlap(

                df1[c1],

                df2[c2]

            )



            confidence=(

                name_score*0.3

                +

                data_score*0.7

            )



            # reject weak fake matches

            if confidence>0.40:


                results.append(

                    {

                    "file1_column":c1,

                    "file2_column":c2,

                    "column_similarity":round(name_score,2),

                    "value_overlap":round(data_score,2),

                    "confidence":round(confidence,2)

                    }

                )





    return sorted(

        results,

        key=lambda x:x["confidence"],

        reverse=True

    )






# ==================================================
# MERGE PREVIEW
# ==================================================

def merge_preview(df1,df2):


    df1=normalize_columns(df1)

    df2=normalize_columns(df2)


    matches=find_matches(
        df1,
        df2
    )


    if matches:

        return matches[0],matches



    return (

        {

        "message":
        "No reliable automatic match found. Use manual merge."

        },

        []

    )








# ==================================================
# MANUAL MERGE
# ==================================================

def merge_on_columns(
    df1,
    df2,
    left_col,
    right_col,
    how="inner"
):


    df1=normalize_columns(df1)

    df2=normalize_columns(df2)



    left_col=(
        left_col.lower()
        .replace(" ","_")
    )


    right_col=(
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






# ==================================================
# SMART AI MERGE
# ==================================================

def smart_ai_merge(
    df1,
    df2,
    threshold=0.45,
    how="inner",
    match_mode="fast"
):


    df1=normalize_columns(df1)

    df2=normalize_columns(df2)



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



    try:


        merged=pd.merge(

            df1,

            df2,


            left_on=best["file1_column"],


            right_on=best["file2_column"],


            how=how

        )


    except Exception:


        merged=pd.DataFrame()



    return (

        merged,

        matches,

        best

    )
