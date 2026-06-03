import pdfplumber
import pandas as pd


def extract_pdf_data(pdf):

    all_tables=[]

    with pdfplumber.open(pdf) as file:

        for page in file.pages:

            tables = page.extract_tables()

            for table in tables:

                df = pd.DataFrame(
                    table[1:], 
                    columns=table[0]
                )

                all_tables.append(df)


    if len(all_tables)>0:
        final_df=pd.concat(
            all_tables,
            ignore_index=True
        )

        return final_df


    return pd.DataFrame()
