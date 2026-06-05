import streamlit as st
import matplotlib.pyplot as plt


def auto_dashboard(df):

    st.subheader(
        "📊 Auto Dashboard"
    )


    try:

        numeric = df.select_dtypes(
            include=["int64","float64"]
        )


        if numeric.empty:

            st.info(
                "No numeric columns available for charts"
            )

            return



        column = st.selectbox(

            "Select column",

            numeric.columns

        )



        fig, ax = plt.subplots()


        ax.hist(

            numeric[column]
            .dropna()

        )


        ax.set_title(

            column

        )


        st.pyplot(fig)



        st.subheader(
            "Correlation Matrix"
        )


        st.dataframe(

            numeric.corr(),

            use_container_width=True

        )



    except Exception as e:


        st.warning(

            f"Dashboard skipped: {e}"

        )
