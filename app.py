import pandas as pd
import streamlit as st


# ================================
# SAFE IMPORTS
# ================================

from modules.cleaner import DataCleaner
from modules.preprocessing import ml_ready
from modules.pdf_reader import extract_pdf_data


from modules.merger import (
    merge_on_columns,
    smart_ai_merge,
    merge_preview
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




# ================================
# PAGE CONFIG
# ================================

st.set_page_config(

    page_title="AI Data Cleaning Platform",

    page_icon="🤖",

    layout="wide"

)




# ================================
# FILE LOADER
# ================================

def load_dataset(file):

    extension = (
        file.name
        .split(".")[-1]
        .lower()
    )


    if extension == "csv":

        return pd.read_csv(file)


    elif extension in ["xlsx","xls"]:

        return pd.read_excel(file)


    elif extension=="pdf":

        return extract_pdf_data(file)


    else:

        raise Exception(
            "Unsupported file type"
        )





# ================================
# QUALITY DASHBOARD
# ================================

def show_quality(df):


    st.subheader(
        "📊 Dataset Quality"
    )


    if dataset_quality:


        result = dataset_quality(df)


        cols = st.columns(
            len(result)
        )


        for i,(k,v) in enumerate(result.items()):


            cols[i].metric(
                k,
                v
            )


    else:


        st.info(
            "Quality module unavailable"
        )





# ================================
# COLUMN AI
# ================================

def show_ai_columns(df):


    st.subheader(
        "🧠 AI Column Understanding"
    )



    if understand_columns:


        data = understand_columns(df)



        st.dataframe(

            pd.DataFrame(

                data.items(),

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






# ================================
# HEADER
# ================================


st.title(
    "🤖 Enterprise AI Data Cleaning System"
)


st.caption(

    "Clean • Understand • Analyze • Merge • Prepare ML Data"

)





# ================================
# SIDEBAR
# ================================


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


if mode=="🧹 Single Dataset Cleaner":



    file = st.file_uploader(

        "Upload Dataset",

        type=[

            "csv",

            "xlsx",

            "xls",

            "pdf"

        ]

    )



    if file is None:


        st.info(
            "Upload a dataset"
        )

        st.stop()




    try:


        df = load_dataset(file)


    except Exception as e:


        st.error(e)

        st.stop()






    show_quality(df)


    show_ai_columns(df)



    st.subheader(
        "Raw Dataset"
    )



    st.dataframe(

        df.head(100),

        use_container_width=True

    )






    if st.button(

        "🚀 Start Cleaning",

        type="primary"

    ):



        try:


            cleaner = DataCleaner(df)



            cleaned = cleaner.clean_all()



            st.success(
                "Cleaning Completed"
            )




            # BEFORE AFTER


            st.subheader(
                "🔍 Before vs After Cleaning"
            )



            st.json(

                {

                    "Before Rows":len(df),

                    "After Rows":len(cleaned),

                    "Rows Removed":len(df)-len(cleaned),


                    "Before Columns":df.shape[1],

                    "After Columns":cleaned.shape[1]

                }

            )





            # EXPLANATION


            st.subheader(

                "📝 Cleaning Explanation"

            )



            if hasattr(
                cleaner,
                "explain_steps"
            ):


                for step in cleaner.explain_steps():


                    st.success(step)



            else:


                st.info(

                    "No cleaning logs"

                )







            show_quality(cleaned)



            st.subheader(
                "Cleaned Dataset"
            )



            st.dataframe(

                cleaned.head(200),

                use_container_width=True

            )







            if auto_dashboard:


                auto_dashboard(
                    cleaned
                )







            try:


                train,test,report = ml_ready(

                    cleaned

                )


                st.subheader(

                    "🤖 ML Preparation"

                )


                st.json(report)



            except Exception as e:


                st.warning(

                    f"ML skipped {e}"

                )







            st.download_button(

                "Download Clean Dataset",

                cleaned.to_csv(index=False),

                "cleaned_dataset.csv",

                "text/csv"

            )




        except Exception as e:



            st.error(

                f"Cleaning failed: {e}"

            )







# =====================================================
# MERGER
# =====================================================


else:



    file1 = st.file_uploader(

        "Upload Dataset 1",

        type=[

            "csv",

            "xlsx",

            "xls",

            "pdf"

        ]

    )



    file2 = st.file_uploader(

        "Upload Dataset 2",

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






    df1 = load_dataset(file1)


    df2 = load_dataset(file2)






    st.subheader(
        "Datasets Uploaded"
    )


    c1,c2 = st.columns(2)



    with c1:


        st.write(
            "Dataset 1"
        )


        st.dataframe(
            df1.head()
        )



    with c2:


        st.write(
            "Dataset 2"
        )


        st.dataframe(
            df2.head()
        )







    merge_mode = st.radio(

        "Merge Method",

        [

            "AI Merge",

            "Manual Merge"

        ]

    )






    if merge_mode=="Manual Merge":



        left = st.selectbox(

            "Dataset 1 Column",

            df1.columns

        )


        right = st.selectbox(

            "Dataset 2 Column",

            df2.columns

        )



    else:



        st.subheader(

            "🔎 AI Merge Preview"

        )



        try:



            preview,all_matches = merge_preview(

                df1,

                df2

            )



            st.json(

                preview

            )



            with st.expander(

                "View all matches"

            ):


                st.dataframe(

                    pd.DataFrame(

                        all_matches

                    )

                )



        except Exception as e:



            st.warning(

                f"Preview unavailable {e}"

            )








    if st.button(

        "🚀 Merge Datasets",

        type="primary"

    ):




        try:



            cleaner1 = DataCleaner(df1)

            cleaner2 = DataCleaner(df2)



            clean1 = cleaner1.clean_all()


            clean2 = cleaner2.clean_all()





            if merge_mode=="Manual Merge":


                merged = merge_on_columns(

                    clean1,

                    clean2,

                    left,

                    right

                )



            else:



                merged,_,_ = smart_ai_merge(

                    clean1,

                    clean2

                )






            if merged.empty:



                st.error(

                    "No matching records found"

                )


                st.stop()








            st.success(

                "Merge Successful"

            )



            show_quality(

                merged

            )



            st.dataframe(

                merged.head(200),

                use_container_width=True

            )






            if auto_dashboard:


                auto_dashboard(

                    merged

                )






            st.download_button(

                "Download Merged Dataset",

                merged.to_csv(index=False),

                "merged_dataset.csv",

                "text/csv"

            )






        except Exception as e:



            st.error(

                f"Merge failed: {e}"

            )
