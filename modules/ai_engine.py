from functools import lru_cache

import numpy as np


def _load_sentence_model():
    try:
        import streamlit as st
        from sentence_transformers import SentenceTransformer

        @st.cache_resource(show_spinner=False)
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


def ai_match_columns(df1, df2, threshold=0.45):
    context1 = create_column_context(df1)
    context2 = create_column_context(df2)

    if not context1 or not context2:
        return []

    model = _load_sentence_model()
    columns1 = list(context1.keys())
    columns2 = list(context2.keys())
    embeddings1 = model.encode(list(context1.values()))
    embeddings2 = model.encode(list(context2.values()))
    embeddings1 = np.asarray(embeddings1, dtype=float)
    embeddings2 = np.asarray(embeddings2, dtype=float)
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
                }
            )

    return matches
