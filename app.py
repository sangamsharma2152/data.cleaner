import pandas as pd
import streamlit as st

from modules.cleaner import DataCleaner
from modules.merger import merge_on_columns, smart_ai_merge
from modules.pdf_reader import extract_pdf_data
from modules.preprocessing import ml_ready


st.set_page_config(
    page_title="AI Data Cleaning Pipeline",
    page_icon="DC",
    layout="wide",
)


def load_uploaded_dataset(uploaded_file):
    extension = uploaded_file.name.split(".")[-1].lower()

    if extension == "csv":
        return pd.read_csv(uploaded_file)
    if extension in {"xlsx", "xls"}:
        return pd.read_excel(uploaded_file)
    if extension == "pdf":
        return extract_pdf_data(uploaded_file)

    raise ValueError("Unsupported file type.")


def report_table(title, report):
    st.write(title)
    st.dataframe(
        pd.DataFrame([report]),
        use_container_width=True,
        hide_index=True,
    )


def display_dataset_summary(label, df):
    missing_values = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    cols = st.columns(4)
    cols[0].metric(f"{label} rows", df.shape[0])
    cols[1].metric(f"{label} columns", df.shape[1])
    cols[2].metric("Missing values", missing_values)
    cols[3].metric("Duplicate rows", duplicate_rows)


st.title("AI Automated Data Cleaning and Dataset Integration")
st.caption(
    "Upload two tabular datasets, clean them, match columns, merge them, and export ML-ready data."
)

with st.sidebar:
    st.header("Pipeline")
    st.write("1. Upload datasets")
    st.write("2. Review raw data")
    st.write("3. Clean and validate")
    st.write("4. Match and merge")
    st.write("5. Prepare ML-ready train/test files")

    st.header("Merge settings")
    match_threshold = st.slider(
        "AI match confidence",
        min_value=0.10,
        max_value=0.95,
        value=0.45,
        step=0.05,
    )
    merge_how = st.selectbox(
        "Merge type",
        options=["outer", "inner", "left", "right"],
        index=0,
    )

st.header("Upload datasets")
uploaded_file_1 = st.file_uploader(
    "First dataset",
    type=["csv", "xlsx", "xls", "pdf"],
)
uploaded_file_2 = st.file_uploader(
    "Second dataset",
    type=["csv", "xlsx", "xls", "pdf"],
)

if not uploaded_file_1 or not uploaded_file_2:
    st.info("Upload two CSV, Excel, or PDF files to start.")
    st.stop()

try:
    df1 = load_uploaded_dataset(uploaded_file_1)
    df2 = load_uploaded_dataset(uploaded_file_2)
except Exception as exc:
    st.error(f"Could not read one of the uploaded files: {exc}")
    st.stop()

if df1.empty or df2.empty:
    st.error("At least one uploaded file did not contain readable table data.")
    st.stop()

st.header("Raw data")
display_dataset_summary("Dataset 1", df1)
display_dataset_summary("Dataset 2", df2)

left, right = st.columns(2)
with left:
    st.subheader("Dataset 1 preview")
    st.dataframe(df1.head(50), use_container_width=True)
with right:
    st.subheader("Dataset 2 preview")
    st.dataframe(df2.head(50), use_container_width=True)

with st.expander("Exploratory data analysis"):
    left, right = st.columns(2)
    with left:
        st.write("Dataset 1 statistics")
        st.dataframe(df1.describe(include="all").transpose(), use_container_width=True)
    with right:
        st.write("Dataset 2 statistics")
        st.dataframe(df2.describe(include="all").transpose(), use_container_width=True)

st.header("Merge controls")
merge_mode = st.radio(
    "Choose merge strategy",
    options=["AI suggested match", "Manual column selection"],
    horizontal=True,
)

manual_left_column = None
manual_right_column = None
if merge_mode == "Manual column selection":
    left, right = st.columns(2)
    with left:
        manual_left_column = st.selectbox("Dataset 1 merge column", df1.columns)
    with right:
        manual_right_column = st.selectbox("Dataset 2 merge column", df2.columns)

if not st.button("Start complete pipeline", type="primary"):
    st.stop()

progress = st.progress(0)

try:
    cleaner1 = DataCleaner(df1)
    cleaner2 = DataCleaner(df2)
    clean_df1 = cleaner1.clean_all()
    clean_df2 = cleaner2.clean_all()
except Exception as exc:
    st.error(f"Cleaning failed: {exc}")
    st.stop()

progress.progress(25)
st.success("Cleaning completed.")

st.subheader("Cleaning report")
left, right = st.columns(2)
with left:
    report_table("Dataset 1", cleaner1.get_report())
with right:
    report_table("Dataset 2", cleaner2.get_report())

try:
    if merge_mode == "Manual column selection":
        merged = merge_on_columns(
            clean_df1,
            clean_df2,
            manual_left_column,
            manual_right_column,
            how=merge_how,
        )
        matches = []
        selected_match = {
            "file1_column": manual_left_column,
            "file2_column": manual_right_column,
            "confidence": "manual",
        }
    else:
        merged, matches, selected_match = smart_ai_merge(
            clean_df1,
            clean_df2,
            threshold=match_threshold,
            how=merge_how,
        )
except Exception as exc:
    st.error(f"Merging failed: {exc}")
    st.stop()

progress.progress(55)

st.subheader("Column matching")
if matches:
    st.write("AI suggested matches")
    st.dataframe(pd.DataFrame(matches), use_container_width=True, hide_index=True)
elif selected_match and selected_match["confidence"] == "manual":
    st.info("Manual merge columns were used.")
else:
    st.warning("No confident matching column was found. Generated IDs were used.")

if selected_match:
    st.write("Selected merge rule")
    st.json(selected_match)

st.subheader("Merged dataset")
display_dataset_summary("Merged", merged)
st.dataframe(merged.head(100), use_container_width=True)

progress.progress(75)

try:
    train, test, prep_report = ml_ready(merged)
except Exception as exc:
    st.error(f"ML preparation failed: {exc}")
    st.stop()

progress.progress(95)

st.subheader("ML preparation report")
st.dataframe(pd.DataFrame([prep_report]), use_container_width=True, hide_index=True)

left, right = st.columns(2)
with left:
    st.write("Training data")
    st.dataframe(train.head(100), use_container_width=True)
with right:
    st.write("Testing data")
    st.dataframe(test.head(100), use_container_width=True)

st.download_button(
    label="Download training CSV",
    data=train.to_csv(index=False),
    file_name="train_cleaned_ml_ready.csv",
    mime="text/csv",
)
st.download_button(
    label="Download testing CSV",
    data=test.to_csv(index=False),
    file_name="test_cleaned_ml_ready.csv",
    mime="text/csv",
)
st.download_button(
    label="Download merged CSV",
    data=merged.to_csv(index=False),
    file_name="merged_cleaned_dataset.csv",
    mime="text/csv",
)

progress.progress(100)
st.success("Complete data pipeline finished successfully.")
