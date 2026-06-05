import pandas as pd


def understand_columns(df):

    meanings = {}


    for col in df.columns:

        name = col.lower()


        if "id" in name:

            meaning="Unique Identifier"

        elif "name" in name:

            meaning="Person/Product Name"

        elif "price" in name or "amount" in name:

            meaning="Financial Value"

        elif "date" in name or "dob" in name:

            meaning="Date Information"

        elif "ram" in name:

            meaning="Memory Specification"

        elif "email" in name:

            meaning="Email Address"

        elif "phone" in name:

            meaning="Phone Number"

        else:

            if pd.api.types.is_numeric_dtype(df[col]):

                meaning="Numerical Feature"

            else:

                meaning="Categorical/Text Feature"



        meanings[col]=meaning


    return meanings
    
