import pandas as pd
from difflib import SequenceMatcher



# ---------------- SIMILARITY ENGINE ----------------

def similarity(a, b):

    return SequenceMatcher(
        None,
        str(a).lower(),
        str(b).lower()
    ).ratio()



# ---------------- VALUE CLEANING ----------------

def clean_value(value):

    value = str(value).lower()

    replacements = [
        "gb",
        "mb",
        "₹",
        "$",
        ","
    ]


    for r in replacements:
        value = value.replace(r,"")


    return value.strip()



# ---------------- NORMALIZER ----------------

def normalize_dataframe(df):

    df = df.copy()


    df.columns = (

        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ","_")

    )


    for col in df.columns:

        df[col] = (
            df[col]
            .astype(str)
            .apply(clean_value)
        )


    return df



# ---------------- FIND BEST MATCH ----------------

def find_best_column_match(df1, df2):

    matches=[]


    for c1 in df1.columns:

        for c2 in df2.columns:


            name_score = similarity(
                c1,
                c2
            )


            value_score = 0


            values1 = (
                df1[c1]
                .dropna()
                .head(20)
            )

            values2 = (
                df2[c2]
                .dropna()
                .head(20)
            )


            scores=[]


            for a in values1:

                for b in values2:

                    scores.append(
                        similarity(a,b)
                    )


            if scores:

                value_score=max(scores)



            final = (

                name_score*0.5
                +
                value_score*0.5

            )



            matches.append(

                {

                "file1_column":c1,

                "file2_column":c2,

                "confidence":round(final,2)

                }

            )



    return sorted(

        matches,

        key=lambda x:x["confidence"],

        reverse=True

    )





# ---------------- MANUAL MERGE ----------------

def merge_on_columns(
        df1,
        df2,
        left_col,
        right_col,
        how="outer"
):


    df1 = normalize_dataframe(df1)
    df2 = normalize_dataframe(df2)


    left_col = left_col.lower().replace(" ","_")
    right_col = right_col.lower().replace(" ","_")


    merged = pd.merge(

        df1,
        df2,

        left_on=left_col,

        right_on=right_col,

        how=how

    )


    return merged




# ---------------- AI MERGE ----------------

def smart_ai_merge(

        df1,
        df2,
        threshold=0.45,
        how="outer",
        match_mode="fast"

):


    df1 = normalize_dataframe(df1)

    df2 = normalize_dataframe(df2)



    matches = find_best_column_match(

        df1,
        df2

    )


    if len(matches)==0:

        raise Exception(
            "No matching columns found"
        )



    best = matches[0]



    merged = pd.merge(

        df1,

        df2,


        left_on=best["file1_column"],

        right_on=best["file2_column"],


        how="outer"

    )



    return (

        merged,

        matches,

        best

    )
