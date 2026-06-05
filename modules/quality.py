def dataset_quality(df):


    total=df.shape[0]*df.shape[1]


    if total==0:

        return {
            "score":0
        }


    missing=df.isna().sum().sum()


    duplicate=df.duplicated().sum()



    completeness=100-(missing/total*100)


    uniqueness=100


    if len(df)>0:

        uniqueness=100-(duplicate/len(df)*100)



    score=(

        completeness*0.7

        +

        uniqueness*0.3

    )



    return {


        "Quality Score":round(score,2),

        "Completeness":round(completeness,2),

        "Uniqueness":round(uniqueness,2),

        "Missing Values":int(missing),

        "Duplicates":int(duplicate)

    }
