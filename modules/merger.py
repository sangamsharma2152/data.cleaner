import hashlib

import pandas as pd

from modules.ai_engine import ai_match_columns


def generate_ai_id(df):
    result = df.copy()
    important = result.columns[: min(3, len(result.columns))]

    if len(important) == 0:
        result["AI_GENERATED_ID"] = []
        return result

    combined = result[important].astype(str).agg("_".join, axis=1)
    result["AI_GENERATED_ID"] = [
        hashlib.md5(value.encode("utf-8")).hexdigest()[:10] for value in combined
    ]
    return result


def merge_on_columns(df1, df2, left_column, right_column, how="outer"):
    if left_column not in df1.columns:
        raise ValueError(f"{left_column} is not present in the first dataset.")
    if right_column not in df2.columns:
        raise ValueError(f"{right_column} is not present in the second dataset.")

    return pd.merge(
        df1,
        df2,
        left_on=left_column,
        right_on=right_column,
        how=how,
        suffixes=("_dataset1", "_dataset2"),
    )


def smart_ai_merge(df1, df2, threshold=0.45, how="outer", match_mode="fast"):
    matches = ai_match_columns(df1, df2, threshold=threshold, mode=match_mode)

    if matches:
        best = max(matches, key=lambda item: item["confidence"])
        merged = merge_on_columns(
            df1,
            df2,
            best["file1_column"],
            best["file2_column"],
            how=how,
        )
        return merged, matches, best

    df1_with_id = generate_ai_id(df1)
    df2_with_id = generate_ai_id(df2)
    merged = pd.merge(
        df1_with_id,
        df2_with_id,
        on="AI_GENERATED_ID",
        how=how,
        suffixes=("_dataset1", "_dataset2"),
    )
    return merged, [], None
