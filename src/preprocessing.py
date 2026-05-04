import numpy as np

def clean_column_names(df):

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df

def clean_string_values(df):

    string_columns = df.select_dtypes(include=['object']).columns

    for column in string_columns:

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
        )

        df[column] = df[column].replace(['NA', 'nan', 'None', ''],np.nan)

    return df