import pandas as pd


def extract_pdf_data(file):

    try:

        import pdfplumber


        tables=[]


        with pdfplumber.open(file) as pdf:


            for page in pdf.pages:


                table = page.extract_table()


                if table:


                    df = pd.DataFrame(

                        table[1:],

                        columns=table[0]

                    )


                    tables.append(df)



        if tables:


            return pd.concat(

                tables,

                ignore_index=True

            )



        return pd.DataFrame()



    except ImportError:


        raise Exception(

            "PDF support missing. Install pdfplumber."

        )



    except Exception as e:


        raise Exception(

            f"PDF reading failed: {e}"

        )
