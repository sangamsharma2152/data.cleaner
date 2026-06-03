import streamlit as st
import pandas as pd


from modules.pdf_reader import extract_pdf_data

from modules.cleaner import DataCleaner

from modules.merger import merge_files

from modules.preprocessing import ml_ready



st.set_page_config(
    page_title="AI Data Cleaning Pipeline",
    layout="wide"
)



st.title(
"AI Automatic Data Cleaning & ML Preparation System"
)



st.write(
"""
Upload two raw PDF datasets.
The system will clean, match, merge,
and prepare ML ready data.
"""
)



pdf1=st.file_uploader(
"Upload First PDF",
type="pdf"
)



pdf2=st.file_uploader(
"Upload Second PDF",
type="pdf"
)



if pdf1 and pdf2:


    df1=extract_pdf_data(pdf1)

    df2=extract_pdf_data(pdf2)



    st.subheader("Raw Dataset 1")

    st.dataframe(df1)



    st.subheader("Raw Dataset 2")

    st.dataframe(df2)



    if st.button(
        "START COMPLETE PIPELINE"
    ):


        # Cleaning

        cleaner1=DataCleaner(df1)

        cleaner2=DataCleaner(df2)


        df1=cleaner1.clean_all()

        df2=cleaner2.clean_all()



        st.success(
            "Cleaning Completed"
        )



        # Merge

        merged=merge_files(
            df1,
            df2
        )


        st.subheader(
            "Merged Data"
        )

        st.dataframe(
            merged
        )



        # ML READY

        train,test=ml_ready(
            merged
        )


        st.success(
            "Machine Learning Dataset Created"
        )



        st.write(
        "Training Dataset"
        )

        st.dataframe(
            train
        )



        csv=train.to_csv(
            index=False
        )


        st.download_button(

            "Download Final Dataset",

            csv,

            "ML_ready_data.csv"

        )
