import pandas as pd
import streamlit as st


from modules.cleaner import DataCleaner
from modules.merger import (
    merge_on_columns,
    smart_ai_merge,
    merge_preview
)

from modules.preprocessing import ml_ready
from modules.pdf_reader import extract_pdf_data


from modules.ai_analyzer import understand_columns
from modules.quality import dataset_quality
from modules.dashboard import auto_dashboard



st.set_page_config(

    page_title="AI Data Platform",

    page_icon="🤖",

    layout="wide"

)




# =================================================
# LOAD DATA
# =================================================


def load_file(file):


    ext=file.name.split(".")[-1]


    if ext=="csv":

        return pd.read_csv(file)



    elif ext in ["xlsx","xls"]:

        return pd.read_excel(file)



    elif ext=="pdf":

        return extract_pdf_data(file)



    else:

        raise Exception(
            "Unsupported File"
        )




# =================================================
# UI HELPERS
# =================================================


def quality_ui(df):


    q=dataset_quality(df)



    cols=st.columns(4)



    for i,(k,v) in enumerate(q.items()):


        cols[i%4].metric(

            k,

            v

        )





def column_ai_ui(df):


    result=understand_columns(df)



    st.dataframe(

        pd.DataFrame(

            result.items(),

            columns=[

                "Column",

                "AI Meaning"

            ]

        ),

        use_container_width=True

    )






# =================================================
# APP
# =================================================


st.title(
    "🤖 Enterprise AI Data Cleaning Platform"
)



mode=st.sidebar.radio(

    "Select Mode",

    [

        "Single Dataset Cleaner",

        "AI Dataset Merger"

    ]

)




# =================================================
# SINGLE CLEANER
# =================================================


if mode=="Single Dataset Cleaner":


    file=st.file_uploader(

        "Upload Dataset",

        type=[
            "csv",
            "xlsx",
            "xls",
            "pdf"
        ]

    )


    if not file:

        st.stop()



    df=load_file(file)



    st.header("Dataset Quality")


    quality_ui(df)



    st.header("AI Column Understanding")


    column_ai_ui(df)




    st.dataframe(
        df.head()
    )




    if st.button(
        "Clean Dataset 🚀"
    ):



        cleaner=DataCleaner(df)



        cleaned=cleaner.clean_all()



        st.success(
            "Cleaning Complete"
        )




        st.subheader(
            "Before vs After"
        )



        st.json(

            {

            "Before Rows":len(df),

            "After Rows":len(cleaned),

            "Rows Removed":len(df)-len(cleaned)

            }

        )




        st.subheader(
            "Cleaning Explanation"
        )



        for step in cleaner.explain_steps():

            st.info(step)




        st.header(
            "After Cleaning Quality"
        )


        quality_ui(cleaned)



        auto_dashboard(cleaned)




        train,test,report=ml_ready(cleaned)




        st.download_button(

            "Download Cleaned CSV",

            cleaned.to_csv(index=False),

            "cleaned.csv"

        )





# =================================================
# MERGER
# =================================================


else:



    f1=st.file_uploader(

        "Dataset 1",

        type=[
            "csv",
            "xlsx",
            "xls"
        ]

    )



    f2=st.file_uploader(

        "Dataset 2",

        type=[
            "csv",
            "xlsx",
            "xls"
        ]

    )



    if not f1 or not f2:

        st.stop()



    df1=load_file(f1)

    df2=load_file(f2)



    cleaner1=DataCleaner(df1)

    cleaner2=DataCleaner(df2)



    clean1=cleaner1.clean_all()

    clean2=cleaner2.clean_all()




    st.header(
        "AI Merge Preview"
    )



    best,all_matches=merge_preview(

        clean1,

        clean2

    )




    st.json(best)




    with st.expander(

        "All Possible Matches"

    ):


        st.dataframe(

            pd.DataFrame(all_matches)

        )





    if st.button(

        "Approve and Merge"

    ):



        merged,matches,selected=smart_ai_merge(

            clean1,

            clean2

        )




        if merged.empty:


            st.error(

                "No reliable merge found. Use manual merge."

            )


            st.stop()




        st.success(
            "Merge Completed"
        )



        quality_ui(
            merged
        )



        auto_dashboard(
            merged
        )




        st.download_button(

            "Download Merged Dataset",

            merged.to_csv(index=False),

            "merged.csv"

        )
