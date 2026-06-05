import pandas as pd
from difflib import SequenceMatcher


def similarity(a, b):
    return SequenceMatcher(
        None,
        str(a).lower(),
        str(b).lower()
    ).ratio()



def normalize_text(value):

    value = str(value).lower()

    replacements = [
        "gb",
        "mb",
        "$",
        "₹",
        ","
    ]

    for item in replacements:
        value = value.replace(item, "")

    return value.strip()



def normalize_dataframe(df):

    df = df.copy()

    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
    )

    for col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .apply(normalize_text)
        )

    return df



def calculate_overlap(a, b):

    first = set(
        a.dropna()
        .astype(str)
        .head(200)
    )

    second = set(
        b.dropna()
        .astype(str)
        .head(200)
    )

    if len(first) == 0:
        return 0

    return len(
        first.intersection(second)
    ) / len(first)




def find_matches(df1, df2):

    matches = []

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
                column_score * 0.4
                +
                value_score * 0.6
            )


            matches.append(
                {
                    "file1_column": c1,
                    "file2_column": c2,
                    "confidence": round(final_score,2),
                    "data_overlap": round(value_score,2)
                }
            )


    return sorted(
        matches,
        key=lambda x:x["confidence"],
        reverse=True
    )




def merge_on_columns(
    df1,
    df2,
    left_col,
    right_col,
    how="inner"
):

    df1 = normalize_dataframe(df1)
    df2 = normalize_dataframe(df2)


    left_col = (
        left_col.lower()
        .replace(" ","_")
    )


    right_col = (
        right_col.lower()
        .replace(" ","_")
    )


    merged = pd.merge(
        df1,
        df2,
        left_on=left_col,
        right_on=right_col,
        how=how
    )


    return merged




def smart_ai_merge(
    df1,
    df2,
    threshold=0.45,
    how="inner",
    match_mode="fast"
):

    df1 = normalize_dataframe(df1)
    df2 = normalize_dataframe(df2)


    matches = find_matches(
        df1,
        df2
    )


    if len(matches) == 0:

        return (
            pd.DataFrame(),
            [],
            {
                "status":"No matches found"
            }
        )


    best = matches[0]


    # don't crash app
    # return information instead

    if best["confidence"] < threshold:

        return (

            pd.DataFrame(),

            matches,

            {
                "status":
                "low confidence",

                "message":
                "Use manual merge",

                "best_guess":
                best
            }

        )



    merged = pd.merge(

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
