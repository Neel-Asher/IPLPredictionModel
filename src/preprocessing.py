import numpy as np

def clean_column_names(df):

    """
    Cleans the column names of the DataFrame by stripping whitespace, converting to lowercase, and replacing spaces with underscores.
    Parameters:
        df (pd.DataFrame): The input DataFrame with original column names.
    Returns:
        pd.DataFrame: The DataFrame with cleaned column names.
    """

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df

def clean_string_values(df):

    """
    Cleans string values in the DataFrame by stripping whitespace and replacing common placeholders for missing values with NaN.
    Parameters:
        df (pd.DataFrame): The input DataFrame containing string values to be cleaned.
    Returns:
        pd.DataFrame: The DataFrame with cleaned string values.
    """

    string_columns = df.select_dtypes(include=['object']).columns

    for column in string_columns:

        df[column] = (df[column].astype(str).str.strip())
        df[column] = df[column].replace(['NA', 'nan', 'None', ''],np.nan)

    return df