import pandas as pd
from difflib import SequenceMatcher



# ================================
# NAME SIMILARITY
# ================================

def similarity(a,b):

    return SequenceMatcher(
        None,
        str(a).lower(),
        str(b).lower()
    ).ratio()





# ================================
# CLEAN COLUMN NAMES ONLY
# ================================

def normalize(df):

    df=df.copy()

    df.columns=(

        df.columns
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(" ","_")

    )

    return df





# ================================
# VALID MERGE KEY CHECK
# ================================

def valid_key(series):


    unique_ratio = (
        series.nunique()
        /
        max(len(series),1)
    )


    # reject true false columns

    if series.nunique() <= 5:

        return False


    # reject columns where everything repeats

    if unique_ratio < 0.2:

        return False


    return True






# ================================
# VALUE MATCHING
# ================================

def overlap(a,b):


    try:

        a=a.dropna().astype(str).str.lower()

        b=b.dropna().astype(str).str.lower()


        sample_a=set(a.head(1000))

        sample_b=set(b.head(1000))


        if not sample_a:

            return 0


        return len(sample_a & sample_b)/len(sample_a)


    except:

        return 0






# ================================
# FIND AI MATCH
# ================================

def find_matches(df1,df2):


    matches=[]


    for c1 in df1.columns:


        if not valid_key(df1[c1]):

            continue



        for c2 in df2.columns:


            if not valid_key(df2[c2]):

                continue




            name_score=similarity(c1,c2)


            value_score=overlap(
                df1[c1],
                df2[c2]
            )




            confidence=(

                name_score*0.5

                +

                value_score*0.5

            )




            if confidence>=0.45:


                matches.append(

                    {

                    "file1_column":c1,

                    "file2_column":c2,

                    "column_similarity":round(name_score,2),

                    "value_overlap":round(value_score,2),

                    "confidence":round(confidence,2)

                    }

                )



    return sorted(

        matches,

        key=lambda x:x["confidence"],

        reverse=True

    )






# ================================
# PREVIEW
# ================================

def merge_preview(df1,df2):


    df1=normalize(df1)

    df2=normalize(df2)



    matches=find_matches(
        df1,
        df2
    )



    if matches:

        return matches[0],matches



    return {

        "message":
        "No safe AI key detected. Use manual merge."

    }, []








# ================================
# AI MERGE
# ================================

def smart_ai_merge(

    df1,
    df2,
    threshold=.45,
    how="inner",
    match_mode="fast"

):


    df1=normalize(df1)

    df2=normalize(df2)



    matches=find_matches(
        df1,
        df2
    )



    if not matches:

        return pd.DataFrame(),[],{}





    key=matches[0]



    try:

        merged=pd.merge(

            df1,

            df2,


            left_on=key["file1_column"],


            right_on=key["file2_column"],


            how=how

        )



    except Exception:


        merged=pd.DataFrame()




    return merged,matches,key








# ================================
# MANUAL MERGE
# ================================

def merge_on_columns(

    df1,
    df2,
    left_col,
    right_col,
    how="inner"

):


    df1=normalize(df1)

    df2=normalize(df2)



    return pd.merge(

        df1,

        df2,


        left_on=left_col.lower(),


        right_on=right_col.lower(),


        how=how

    )
