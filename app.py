import pandas as pd
import streamlit as st

from modules.cleaner import DataCleaner
from modules.merger import merge_on_columns, smart_ai_merge
from modules.pdf_reader import extract_pdf_data
from modules.preprocessing import ml_ready



# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="AI Data Cleaning Platform",
    page_icon="🤖",
    layout="wide"
)



# ==================================================
# LOAD DATA
# ==================================================

def load_uploaded_dataset(file):

    ext = file.name.split(".")[-1].lower()


    if ext == "csv":
        return pd.read_csv(file)


    elif ext in ["xlsx","xls"]:
        return pd.read_excel(file)


    elif ext=="pdf":
        return extract_pdf_data(file)


    else:
        raise Exception(
            "Unsupported file"
        )




# ==================================================
# DISPLAY HELPERS
# ==================================================

def dataset_summary(name,df):

    st.subheader(name)


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




def show_report(report):

    st.dataframe(

        pd.DataFrame([report]),

        use_container_width=True,

        hide_index=True

    )





# ==================================================
# MAIN UI
# ==================================================

st.title(
    "🤖 Enterprise AI Data Cleaner"
)


st.caption(
    "Clean datasets • Merge databases • Prepare ML datasets"
)



with st.sidebar:


    st.header(
        "Choose Pipeline"
    )


    mode = st.radio(

        "Mode",

        [
            "🧹 Single Dataset Cleaner",

            "🔗 AI Dataset Merger"
        ]

    )





# ==================================================
# SINGLE DATASET CLEANER
# ==================================================


if mode=="🧹 Single Dataset Cleaner":



    st.header(
        "Upload Dataset"
    )



    uploaded = st.file_uploader(

        "Upload CSV / Excel / PDF",

        type=[

            "csv",

            "xlsx",

            "xls",

            "pdf"

        ]

    )



    if uploaded is None:

        st.info(
            "Upload a dataset"
        )

        st.stop()



    df = load_uploaded_dataset(
        uploaded
    )



    dataset_summary(

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


        try:


            cleaner = DataCleaner(df)


            cleaned = cleaner.clean_all()



            st.success(
                "Cleaning completed"
            )


            dataset_summary(

                "Cleaned Dataset",

                cleaned

            )



            st.subheader(
                "Cleaning Report"
            )


            show_report(

                cleaner.get_report()

            )



            st.dataframe(

                cleaned.head(200),

                use_container_width=True

            )



            # ML


            train,test,report = ml_ready(

                cleaned

            )



            st.subheader(

                "ML Preparation"

            )


            show_report(
                report
            )



            st.download_button(

                "Download Clean Dataset",

                cleaned.to_csv(index=False),

                "cleaned_dataset.csv",

                "text/csv"

            )



            st.download_button(

                "Download ML Dataset",

                train.to_csv(index=False),

                "ml_dataset.csv",

                "text/csv"

            )




        except Exception as e:


            st.error(

                f"Error: {e}"

            )





# ==================================================
# AI DATASET MERGER
# ==================================================



else:



    with st.sidebar:


        st.header(
            "Merge Settings"
        )



        confidence = st.slider(

            "AI Match Confidence",

            0.10,

            0.95,

            0.30,

            step=0.05

        )



        merge_type = st.selectbox(

            "Merge Type",

            [

                "inner",

                "outer",

                "left",

                "right"

            ]

        )




    st.header(
        "Upload Datasets"
    )



    file1 = st.file_uploader(

        "Dataset 1",

        type=[

            "csv",

            "xlsx",

            "xls",

            "pdf"

        ]

    )


    file2 = st.file_uploader(

        "Dataset 2",

        type=[

            "csv",

            "xlsx",

            "xls",

            "pdf"

        ]

    )



    if not file1 or not file2:


        st.info(

            "Upload two datasets"

        )


        st.stop()





    df1 = load_uploaded_dataset(

        file1

    )


    df2 = load_uploaded_dataset(

        file2

    )



    c1,c2 = st.columns(2)



    with c1:


        dataset_summary(

            "Dataset 1",

            df1

        )


        st.dataframe(

            df1.head(),

            use_container_width=True

        )



    with c2:


        dataset_summary(

            "Dataset 2",

            df2

        )


        st.dataframe(

            df2.head(),

            use_container_width=True

        )




    merge_mode = st.radio(

        "Merge Method",

        [

            "AI Merge",

            "Manual Merge"

        ],

        horizontal=True

    )




    left_col=None

    right_col=None




    if merge_mode=="Manual Merge":



        a,b = st.columns(2)


        with a:


            left_col = st.selectbox(

                "Dataset 1 Column",

                df1.columns

            )



        with b:


            right_col = st.selectbox(

                "Dataset 2 Column",

                df2.columns

            )






    if st.button(

        "🚀 Start Merge Pipeline",

        type="primary"

    ):


        try:


            progress = st.progress(0)



            cleaner1 = DataCleaner(

                df1

            )


            cleaner2 = DataCleaner(

                df2

            )



            clean1 = cleaner1.clean_all()


            clean2 = cleaner2.clean_all()



            progress.progress(30)



            st.success(

                "Cleaning finished"

            )





            if merge_mode=="Manual Merge":


                merged = merge_on_columns(

                    clean1,

                    clean2,

                    left_col,

                    right_col,

                    how=merge_type

                )


                selected={

                    "mode":"manual",

                    "column1":left_col,

                    "column2":right_col

                }



                matches=[]



            else:


                merged,matches,selected = smart_ai_merge(

                    clean1,

                    clean2,

                    threshold=confidence,

                    how=merge_type

                )





            progress.progress(70)



            st.subheader(

                "Merge Decision"

            )



            st.json(

                selected

            )



            if matches:


                with st.expander(

                    "AI Match Details"

                ):


                    st.dataframe(

                        pd.DataFrame(matches),

                        use_container_width=True

                    )





            if merged.empty:



                st.warning(

                    """
Merge completed but no matching rows found.

The datasets probably do not share a common key.

Try Manual Merge.
                    """

                )


                st.stop()






            st.subheader(

                "Merged Dataset"

            )



            dataset_summary(

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



            st.subheader(

                "ML Ready Report"

            )


            show_report(

                report

            )




            st.download_button(

                "Download Merged Dataset",

                merged.to_csv(index=False),

                "merged_dataset.csv",

                "text/csv"

            )



            st.download_button(

                "Download ML Dataset",

                train.to_csv(index=False),

                "ml_ready_dataset.csv",

                "text/csv"

            )



            progress.progress(100)



            st.success(

                "Pipeline completed successfully 🚀"

            )





        except Exception as e:


            st.error(

                f"Pipeline failed: {e}"

            )
