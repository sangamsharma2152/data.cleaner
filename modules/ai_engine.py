from difflib import SequenceMatcher
from functools import lru_cache

import numpy as np


def _normalize_column_name(name):
    return "".join(char.lower() for char in str(name) if char.isalnum())


def _name_similarity(left, right):
    left_normalized = _normalize_column_name(left)
    right_normalized = _normalize_column_name(right)

    if not left_normalized or not right_normalized:
        return 0.0
    if left_normalized == right_normalized:
        return 1.0
    if left_normalized in right_normalized or right_normalized in left_normalized:
        return 0.92

    return SequenceMatcher(None, left_normalized, right_normalized).ratio()


def _load_sentence_model():
    try:
        import streamlit as st
        from sentence_transformers import SentenceTransformer

        @st.cache_resource(show_spinner="Loading deep AI matcher...")
        def cached_model():
            return SentenceTransformer("all-MiniLM-L6-v2")

        return cached_model()
    except Exception:
        return _load_sentence_model_without_streamlit()


@lru_cache(maxsize=1)
def _load_sentence_model_without_streamlit():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def create_column_context(df):
    result = {}

    for col in df.columns:
        samples = df[col].dropna().astype(str).head(5).tolist()
        result[col] = f"Column name: {col}. Examples: {' '.join(samples)}"

    return result


def fast_match_columns(df1, df2, threshold=0.75):
    matches = []

    for col1 in df1.columns:
        best_column = None
        best_score = 0.0

        for col2 in df2.columns:
            score = _name_similarity(col1, col2)

            if score > best_score:
                best_column = col2
                best_score = score

        if best_column is not None and best_score >= threshold:
            matches.append(
                {
                    "file1_column": col1,
                    "file2_column": best_column,
                    "confidence": round(best_score, 4),
                    "method": "fast",
                }
            )

    return matches


def semantic_match_columns(df1, df2, threshold=0.45):
    context1 = create_column_context(df1)
    context2 = create_column_context(df2)

    if not context1 or not context2:
        return []

    model = _load_sentence_model()
    columns1 = list(context1.keys())
    columns2 = list(context2.keys())
    embeddings1 = np.asarray(model.encode(list(context1.values())), dtype=float)
    embeddings2 = np.asarray(model.encode(list(context2.values())), dtype=float)
    embeddings1 = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)
    embeddings2 = embeddings2 / np.linalg.norm(embeddings2, axis=1, keepdims=True)
    scores = embeddings1 @ embeddings2.T

    matches = []
    for row_index, col in enumerate(columns1):
        best_index = int(np.argmax(scores[row_index]))
        confidence = float(scores[row_index][best_index])

        if confidence >= threshold:
            matches.append(
                {
                    "file1_column": col,
                    "file2_column": columns2[best_index],
                    "confidence": round(confidence, 4),
                    "method": "deep_ai",
                }
            )

    return matches


def ai_match_columns(df1, df2, threshold=0.45, mode="fast"):
    if mode == "deep_ai":
        return semantic_match_columns(df1, df2, threshold=threshold)

    fast_threshold = max(0.55, threshold)
    return fast_match_columns(df1, df2, threshold=fast_threshold)
