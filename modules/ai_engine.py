from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np



model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)




def create_column_context(df):


    result={}


    for col in df.columns:


        samples = (

            df[col]
            .dropna()
            .astype(str)
            .head(5)
            .tolist()
        )


        text = (
            "Column name: "
            + col
            +
            " Examples: "
            +
            " ".join(samples)
        )


        result[col]=text


    return result




def ai_match_columns(
        df1,
        df2
):


    context1=create_column_context(df1)

    context2=create_column_context(df2)



    matches=[]


    for c1,text1 in context1.items():


        emb1=model.encode(
            text1
        )


        best=None

        score=0



        for c2,text2 in context2.items():


            emb2=model.encode(
                text2
            )


            similarity = cosine_similarity(

                [emb1],

                [emb2]

            )[0][0]



            if similarity>score:

                score=similarity

                best=c2



        if score > 0.45:


            matches.append(
                {
                    "file1_column":c1,
                    "file2_column":best,
                    "confidence":score
                }
            )


    return matches
