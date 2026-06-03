import streamlit as st
import pandas as pd
import numpy as np


# -------------------------------
# Import Custom Modules
# -------------------------------

from modules.pdf_reader import extract_pdf_data
from modules.cleaner import DataCleaner
from modules.merger import smart_ai_merge
from modules.preprocessing import ml_ready



# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(

    page_title="AI Data Cleaning Pipeline",

    page_icon="🤖",

    layout="wide"

)



# -------------------------------
# Application Title
# -------------------------------


st.title(
    "🤖 AI Automated Data Cleaning & Dataset Integration System"
)



st.write(
"""
This application automatically performs:

✔ Raw Data Extraction  
✔ Data Collection  
✔ Exploratory Data Analysis  
✔ Data Cleaning  
✔ Missing Value Handling  
✔ Duplicate Removal  
✔ Outlier Handling  
✔ AI Column Understanding  
✔ AI Dataset Matching  
✔ Automatic Unique ID Generation  
✔ Dataset Merging  
✔ Encoding  
✔ Feature Engineering  
✔ Feature Selection  
✔ Scaling  
✔ Train-Test Split  

Upload two PDF datasets and AI will prepare them for Machine Learning.
"""
)



# -------------------------------
# Sidebar
# -------------------------------


st.sidebar.title(
    "Pipeline Steps"
)


steps=[
    "PDF Extraction",
    "EDA",
    "Cleaning",
    "Missing Values",
    "Duplicates",
    "Outliers",
    "AI Matching",
    "AI Merge",
    "Encoding",
    "Scaling",
    "ML Ready"
]


for step in steps:

    st.sidebar.success(step)





# -------------------------------
# Upload Files
# -------------------------------


st.header(
    "Upload Raw PDF Files"
)


pdf1 = st.file_uploader(

    "Upload First PDF Dataset",

    type=["pdf"]

)



pdf2 = st.file_uploader(

    "Upload Second PDF Dataset",

    type=["pdf"]

)





# -------------------------------
# Main Pipeline
# -------------------------------



if pdf1 and pdf2:



    st.info(
        "Extracting data from PDF files..."
    )


    # Extract PDF tables

    df1 = extract_pdf_data(pdf1)

    df2 = extract_pdf_data(pdf2)




    if df1.empty or df2.empty:


        st.error(
            "Could not extract tables from PDF"
        )


        st.stop()





    # ---------------------------
    # RAW DATA DISPLAY
    # ---------------------------


    st.subheader(
        "Raw Dataset 1"
    )


    st.dataframe(
        df1,
        use_container_width=True
    )



    st.subheader(
        "Raw Dataset 2"
    )


    st.dataframe(
        df2,
        use_container_width=True
    )





    # ---------------------------
    # EDA
    # ---------------------------


    with st.expander(
        "Exploratory Data Analysis"
    ):


        col1,col2=st.columns(2)


        with col1:


            st.write(
                "Dataset 1 Shape"
            )

            st.write(
                df1.shape
            )


            st.write(
                df1.describe()
            )



        with col2:


            st.write(
                "Dataset 2 Shape"
            )


            st.write(
                df2.shape
            )


            st.write(
                df2.describe()
            )






    # ---------------------------
    # START BUTTON
    # ---------------------------



    if st.button(
        "🚀 START COMPLETE AI PIPELINE"
    ):




        progress = st.progress(0)





        # -----------------------
        # CLEANING
        # -----------------------


        st.subheader(
            "Cleaning Data"
        )



        cleaner1=DataCleaner(df1)

        cleaner2=DataCleaner(df2)



        df1 = cleaner1.clean_all()


        df2 = cleaner2.clean_all()



        progress.progress(20)



        st.success(
            "✔ Data Cleaning Completed"
        )








        # -----------------------
        # AI MERGING
        # -----------------------



        st.subheader(
            "AI Understanding Columns"
        )



        merged , matches = smart_ai_merge(

            df1,

            df2

        )



        progress.progress(50)





        if len(matches)>0:



            st.success(

                "AI Found Matching Attributes"

            )



            match_df=pd.DataFrame(

                matches

            )



            st.dataframe(

                match_df,

                use_container_width=True

            )



        else:


            st.warning(

                "No common ID found. AI generated unique IDs."

            )






        # -----------------------
        # MERGED DATA
        # -----------------------



        st.subheader(

            "Merged Dataset"

        )



        st.dataframe(

            merged,

            use_container_width=True

        )



        st.write(

            "Merged Shape:",

            merged.shape

        )



        progress.progress(70)








        # -----------------------
        # MACHINE LEARNING READY
        # -----------------------



        st.subheader(

            "Machine Learning Preparation"

        )



        train , test = ml_ready(

            merged

        )



        progress.progress(90)



        st.success(

            "Dataset Converted Into ML Ready Format"

        )







        col1,col2=st.columns(2)



        with col1:


            st.write(

                "Training Data"

            )



            st.dataframe(

                train,

                use_container_width=True

            )



        with col2:


            st.write(

                "Testing Data"

            )


            st.dataframe(

                test,

                use_container_width=True

            )








        # -----------------------
        # DOWNLOAD
        # -----------------------



        final_csv = train.to_csv(

            index=False

        )



        st.download_button(

            label=

            "⬇ Download ML Ready Dataset",


            data=final_csv,


            file_name=

            "AI_Cleaned_ML_Dataset.csv",


            mime=

            "text/csv"

        )





        progress.progress(100)



        st.balloons()



        st.success(

            "🎉 Complete Data Pipeline Finished Successfully"

        )



else:


    st.info(

        "Upload two PDF files to start the AI pipeline"

    )
