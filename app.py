import pandas as pd
import streamlit as st

from modules.cleaner import DataCleaner
from modules.merger import merge_on_columns, smart_ai_merge
from modules.pdf_reader import extract_pdf_data
from modules.preprocessing import ml_ready


# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Data Cleaner Pro",
    page_icon="🤖",
    layout="wide"
)


# ---------------- HELPERS ----------------

def load_uploaded_dataset(uploaded_file):

    ext = uploaded_file.name.split(".")[-1].lower()

    if ext == "csv":
        return pd.read_csv(uploaded_file)

    elif ext in ["xlsx","xls"]:
        return pd.read_excel(uploaded_file)

    elif ext == "pdf":
        return extract_pdf_data(uploaded_file)

    else:
        raise Exception("Unsupported file")


def show_summary(title, df):

    st.subheader(title)

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Rows",
        df.shape[0]
    )

    c2.metric(
        "Columns",
        df.shape[1]
    )

    c3.metric(
        "Missing",
        int(df.isna().sum().sum())
    )

    c4.metric(
        "Duplicates",
        int(df.duplicated().sum())
    )



def cleaning_report(report):

    st.dataframe(
        pd.DataFrame([report]),
        hide_index=True,
        use_container_width=True
    )


# fixes RAM 8GB vs 8 problem

def normalize_columns(df):

    for col in df.columns:

        try:
            df[col] = (
                df[col]
                .astype(str)
                .str.lower()
                .str.replace("gb","")
                .str.replace(",","")
                .str.strip()
            )

        except:
            pass

    return df



# ---------------- UI ----------------


st.title(
    "🤖 Enterprise AI Data Cleaning System"
)


st.caption(
    "Clean datasets, merge databases and generate ML ready files"
)


with st.sidebar:

    st.header("Choose Pipeline")

    app_mode = st.radio(

        "Mode",

        [
            "🧹 Single Dataset Cleaner",
            "🔗 AI Dataset Merger"
        ]

    )




# =====================================================
# SINGLE FILE CLEANING
# =====================================================


if app_mode=="🧹 Single Dataset Cleaner":


    st.header(
        "Upload Single Dataset"
    )


    file = st.file_uploader(

        "Upload CSV / Excel / PDF",

        type=[
            "csv",
            "xlsx",
            "xls",
            "pdf"
        ]

    )


    if not file:
        st.stop()



    df = load_uploaded_dataset(file)


    show_summary(
        "Original Dataset",
        df
    )


    st.dataframe(
        df.head(100),
        use_container_width=True
    )



    if st.button(
        "🚀 Clean Dataset",
        type="primary"
    ):


        cleaner = DataCleaner(df)


        cleaned = cleaner.clean_all()


        st.success(
            "Cleaning Completed"
        )


        show_summary(
            "Clean Dataset",
            cleaned
        )


        st.subheader(
            "Cleaning Report"
        )


        cleaning_report(
            cleaner.get_report()
        )


        st.dataframe(
            cleaned.head(200),
            use_container_width=True
        )



        try:

            train,test,prep = ml_ready(cleaned)


            st.subheader(
                "ML Ready Report"
            )


            cleaning_report(prep)


            st.download_button(

                "Download ML Train CSV",

                train.to_csv(index=False),

                "train.csv"

            )


        except:

            st.warning(
                "ML preparation skipped"
            )



        st.download_button(

            "Download Cleaned Dataset",

            cleaned.to_csv(index=False),

            "cleaned_dataset.csv",

            "text/csv"

        )





# =====================================================
# MULTI DATASET MERGING
# =====================================================



else:


    st.header(
        "AI Dataset Merger"
    )


    with st.sidebar:


        threshold = st.slider(

            "AI Match Confidence",

            0.1,
            0.95,
            0.45

        )


        merge_type = st.selectbox(

            "Merge Type",

            [
                "outer",
                "inner",
                "left",
                "right"
            ]

        )



    f1 = st.file_uploader(

        "Dataset 1",

        type=[
            "csv",
            "xlsx",
            "xls",
            "pdf"
        ]

    )


    f2 = st.file_uploader(

        "Dataset 2",

        type=[
            "csv",
            "xlsx",
            "xls",
            "pdf"
        ]

    )


    if not f1 or not f2:

        st.info(
            "Upload two datasets"
        )

        st.stop()



    df1 = load_uploaded_dataset(f1)

    df2 = load_uploaded_dataset(f2)



    col1,col2 = st.columns(2)


    with col1:

        show_summary(
            "Dataset 1",
            df1
        )


        st.dataframe(df1.head())


    with col2:

        show_summary(
            "Dataset 2",
            df2
        )


        st.dataframe(df2.head())



    mode = st.radio(

        "Merge Strategy",

        [
            "AI Auto Merge",
            "Manual Merge"
        ]

    )



    left_col=None

    right_col=None



    if mode=="Manual Merge":


        c1,c2=st.columns(2)


        with c1:

            left_col=st.selectbox(

                "Dataset 1 column",

                df1.columns

            )


        with c2:

            right_col=st.selectbox(

                "Dataset 2 column",

                df2.columns

            )




    if st.button(
        "Start Merge Pipeline",
        type="primary"
    ):


        progress=st.progress(0)



        cleaner1=DataCleaner(df1)

        cleaner2=DataCleaner(df2)



        clean1=cleaner1.clean_all()

        clean2=cleaner2.clean_all()


        clean1=normalize_columns(clean1)

        clean2=normalize_columns(clean2)


        progress.progress(40)


        st.success(
            "Both datasets cleaned"
        )



        try:


            if mode=="Manual Merge":


                merged=merge_on_columns(

                    clean1,
                    clean2,
                    left_col,
                    right_col,
                    how=merge_type

                )

                matches=[]



            else:


                merged,matches,selected=smart_ai_merge(

                    clean1,
                    clean2,

                    threshold=threshold,

                    how=merge_type

                )


        except Exception as e:


            st.error(
                f"Merge Failed: {e}"
            )

            st.stop()



        progress.progress(80)



        st.subheader(
            "Merged Dataset"
        )



        show_summary(
            "Merged",
            merged
        )



        st.dataframe(
            merged.head(200),
            use_container_width=True
        )



        train,test,report = ml_ready(
            merged
        )



        st.download_button(

            "Download Merged Dataset",

            merged.to_csv(index=False),

            "merged_dataset.csv"

        )


        st.download_button(

            "Download ML Dataset",

            train.to_csv(index=False),

            "ml_ready.csv"

        )



        progress.progress(100)


        st.success(
            "Pipeline Completed Successfully 🚀"
        )
