import pandas as pd
import streamlit as st


# =====================================================
# IMPORTS
# =====================================================

from modules.cleaner import DataCleaner
from modules.preprocessing import ml_ready
from modules.pdf_reader import extract_pdf_data


from modules.merger import (
    smart_ai_merge,
    merge_preview,
    merge_on_columns
)


try:
    from modules.ai_analyzer import understand_columns
except:
    understand_columns = None


try:
    from modules.quality import dataset_quality
except:
    dataset_quality = None


try:
    from modules.dashboard import auto_dashboard
except:
    auto_dashboard = None



# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Data Cleaning Platform",
    page_icon="🤖",
    layout="wide"
)



# =====================================================
# LOAD FILE
# =====================================================

def load_dataset(file):

    ext = file.name.split(".")[-1].lower()


    if ext == "csv":

        return pd.read_csv(file)


    elif ext in ["xlsx","xls"]:

        return pd.read_excel(file)


    elif ext == "pdf":

        return extract_pdf_data(file)


    else:

        raise Exception("Unsupported file")





# =====================================================
# QUALITY UI
# =====================================================

def show_quality(df):

    st.subheader(
        "📊 Dataset Quality Score"
    )


    if dataset_quality:

        result = dataset_quality(df)

        cols = st.columns(
            len(result)
        )


        for i,(key,value) in enumerate(
            result.items()
        ):

            cols[i].metric(
                key,
                value
            )


    else:

        st.info(
            "Quality module unavailable"
        )





# =====================================================
# COLUMN AI UI
# =====================================================

def show_column_ai(df):


    st.subheader(
        "🧠 AI Column Understanding"
    )


    if understand_columns:


        result = understand_columns(df)


        st.dataframe(

            pd.DataFrame(
                result.items(),
                columns=[
                    "Column",
                    "Detected Meaning"
                ]
            ),

            use_container_width=True

        )


    else:


        st.warning(
            "AI analyzer unavailable"
        )





# =====================================================
# APP HEADER
# =====================================================


st.title(
    "🤖 Enterprise AI Data Cleaning Platform"
)


st.caption(
    "Clean • Analyze • Understand • Merge • Prepare ML Data"
)





# =====================================================
# SIDEBAR
# =====================================================


mode = st.sidebar.radio(

    "Choose Pipeline",

    [

        "🧹 Single Dataset Cleaner",

        "🔗 AI Dataset Merger"

    ]

)





# =====================================================
# SINGLE DATASET CLEANER
# =====================================================


if mode == "🧹 Single Dataset Cleaner":


    uploaded = st.file_uploader(

        "Upload Dataset",

        type=[
            "csv",
            "xlsx",
            "xls",
            "pdf"
        ]

    )


    if uploaded is None:

        st.stop()



    try:

        df = load_dataset(uploaded)


    except Exception as e:


        st.error(e)

        st.stop()



    show_quality(df)


    show_column_ai(df)



    st.subheader(
        "Raw Dataset"
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
                "Cleaning Completed"
            )



            st.subheader(
                "Before vs After Cleaning"
            )


            st.json(

                {

                "Before Rows":len(df),

                "After Rows":len(cleaned),

                "Rows Removed":
                len(df)-len(cleaned),

                "Before Columns":
                df.shape[1],

                "After Columns":
                cleaned.shape[1]

                }

            )




            st.subheader(
                "Cleaning Explanation"
            )



            for step in cleaner.explain_steps():


                st.info(step)





            show_quality(cleaned)





            st.subheader(
                "Cleaned Dataset"
            )



            st.dataframe(

                cleaned.head(200),

                use_container_width=True

            )




            if auto_dashboard:


                auto_dashboard(cleaned)






            try:


                train,test,report = ml_ready(
                    cleaned
                )


                st.subheader(
                    "ML Preparation"
                )


                st.json(report)



            except Exception as e:


                st.warning(
                    f"ML skipped: {e}"
                )






            st.download_button(

                "Download Clean CSV",

                cleaned.to_csv(index=False),

                "cleaned_dataset.csv",

                "text/csv"

            )





        except Exception as e:


            st.error(

                f"Cleaning failed: {e}"

            )







# =====================================================
# AI MERGER
# =====================================================


else:


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






    try:


        df1 = load_dataset(file1)


        df2 = load_dataset(file2)



    except Exception as e:


        st.error(e)


        st.stop()





    st.subheader(
        "Uploaded Data Preview"
    )



    c1,c2 = st.columns(2)



    with c1:

        st.write(
            "Dataset 1"
        )

        st.dataframe(
            df1.head(),
            use_container_width=True
        )


    with c2:

        st.write(
            "Dataset 2"
        )


        st.dataframe(
            df2.head(),
            use_container_width=True
        )





    merge_type = st.radio(

        "Merge Method",

        [

            "AI Merge",

            "Manual Merge"

        ]

    )





    if merge_type=="Manual Merge":


        left_col = st.selectbox(

            "Dataset 1 Column",

            df1.columns

        )


        right_col = st.selectbox(

            "Dataset 2 Column",

            df2.columns

        )





    else:


        st.subheader(
            "🔎 AI Merge Preview"
        )



        try:


            preview,matches = merge_preview(

                df1,

                df2

            )


            st.json(preview)



            with st.expander(
                "View all matches"
            ):


                st.dataframe(

                    pd.DataFrame(matches),

                    use_container_width=True

                )



        except Exception as e:


            st.warning(

                f"Preview failed {e}"

            )






    if st.button(

        "🚀 Merge Datasets",

        type="primary"

    ):


        try:


            # =====================================
            # IMPORTANT FIX:
            # MERGE RAW DATA FIRST
            # THEN CLEAN
            # =====================================


            if merge_type=="Manual Merge":


                merged = merge_on_columns(

                    df1,

                    df2,

                    left_col,

                    right_col

                )



            else:


                merged,_,_ = smart_ai_merge(

                    df1,

                    df2

                )





            if merged.empty:


                st.error(

                    "No valid merge found. Try Manual Merge."

                )


                st.stop()






            cleaner = DataCleaner(

                merged

            )


            merged_clean = (

                cleaner.clean_all()

            )





            st.success(

                "Merge Completed Successfully"

            )





            st.subheader(

                "Merge Cleaning Explanation"

            )



            for step in cleaner.explain_steps():

                st.info(step)






            show_quality(

                merged_clean

            )






            st.subheader(

                "Merged Dataset"

            )



            st.dataframe(

                merged_clean.head(300),

                use_container_width=True

            )






            if auto_dashboard:


                auto_dashboard(

                    merged_clean

                )






            st.download_button(

                "Download Merged CSV",

                merged_clean.to_csv(index=False),

                "merged_dataset.csv",

                "text/csv"

            )





        except Exception as e:


            st.error(

                f"Merge failed: {e}"

            )
