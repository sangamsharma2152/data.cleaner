import pandas as pd
from difflib import SequenceMatcher



# -------------------------
# Similarity
# -------------------------

def similarity(a,b):

    return SequenceMatcher(
        None,
        str(a).lower(),
        str(b).lower()
    ).ratio()



# -------------------------
# Cleaning
# -------------------------

def normalize_value(x):

    x = str(x).lower()

    remove=[
        "gb",
        "mb",
        "$",
        "₹",
        ","
    ]

    for r in remove:
        x=x.replace(r,"")

    return x.strip()



def normalize(df):

    df=df.copy()

    df.columns=(

        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ","_")

    )


    for c in df.columns:

        df[c]=(
            df[c]
            .astype(str)
            .apply(normalize_value)
        )

    return df




# -------------------------
# Check actual row overlap
# -------------------------

def value_overlap(
        s1,
        s2
):

    a=set(
        s1.dropna()
        .astype(str)
        .head(200)
    )

    b=set(
        s2.dropna()
        .astype(str)
        .head(200)
    )


    if len(a)==0:
        return 0


    return len(a.intersection(b))/len(a)





# -------------------------
# AI column detector
# -------------------------

def detect_matches(
        df1,
        df2
):

    results=[]


    for c1 in df1.columns:

        for c2 in df2.columns:


            name=similarity(
                c1,
                c2
            )


            overlap=value_overlap(

                df1[c1],

                df2[c2]

            )


            score=(

                name*0.3

                +

                overlap*0.7

            )


            results.append({

                "file1_column":c1,

                "file2_column":c2,

                "name_similarity":round(name,2),

                "data_overlap":round(overlap,2),

                "confidence":round(score,2)

            })


    return sorted(

        results,

        key=lambda x:x["confidence"],

        reverse=True

    )




# -------------------------
# Manual merge
# -------------------------

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






# -------------------------
# AI MERGE
# -------------------------

def smart_ai_merge(

        df1,
        df2,
        threshold=.50,
        how="inner",
        match_mode="fast"

):


    df1=normalize(df1)

    df2=normalize(df2)



    matches=detect_matches(

        df1,

        df2

    )



    best=matches[0]


    # important check

    if best["data_overlap"] < 0.20:


        raise Exception(

            f"""
AI could not find a real matching column.

Best guess:
{best['file1_column']}
 ↔
{best['file2_column']}

but data similarity was only:
{best['data_overlap']}

Please use manual merge.
            """

        )



    merged=pd.merge(

        df1,

        df2,

        left_on=best["file1_column"],

        right_on=best["file2_column"],

        how=how

    )


    return merged,matches,best
