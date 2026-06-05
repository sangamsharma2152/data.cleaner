from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



def ml_ready(df):


    if df is None or df.empty:


        return (

            df,

            df,

            {

                "status":

                "Dataset empty - ML skipped"

            }

        )




    data = df.copy()



    encoded = 0



    for col in data.columns:



        if data[col].dtype=="object":



            encoder = LabelEncoder()



            data[col] = (

                encoder
                .fit_transform(
                    data[col].astype(str)
                )

            )



            encoded += 1





    if len(data)<5:


        return (

            data,

            data,

            {

                "status":

                "Small dataset - no split",

                "rows":len(data)

            }

        )




    train,test=train_test_split(

        data,

        test_size=.2,

        random_state=42

    )




    report={


        "status":"success",


        "training_rows":len(train),


        "testing_rows":len(test),


        "encoded_columns":encoded

    }



    return train,test,report
