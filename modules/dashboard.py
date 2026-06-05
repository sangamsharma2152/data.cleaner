import streamlit as st
import matplotlib.pyplot as plt


def auto_dashboard(df):


    st.subheader(
        "Automatic Data Dashboard"
    )


    numeric=df.select_dtypes(
        include="number"
    )


    if numeric.empty:

        st.info(
            "No numeric columns available"
        )

        return



    column=st.selectbox(

        "Analyze Column",

        numeric.columns

    )



    fig,ax=plt.subplots()


    ax.hist(
        numeric[column]
    )


    ax.set_title(
        column
    )


    st.pyplot(fig)



    st.subheader(
        "Correlation"
    )


    st.dataframe(

        numeric.corr(),

        use_container_width=True

    )
