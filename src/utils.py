def merge_datasets(deliveries_df, matches_df):

    """
    Merges the deliveries and matches DataFrames on the 'match_id' and 'id' columns, respectively, 
    to create a single DataFrame that contains all relevant information for feature engineering and 
    model training. The function performs a left join to ensure that all deliveries are retained 
    in the merged DataFrame, even if there is no corresponding match information available.
    Parameters:
        deliveries_df (pd.DataFrame): The DataFrame containing delivery-level data.
        matches_df (pd.DataFrame): The DataFrame containing match-level data.
    Returns:
        pd.DataFrame: A merged DataFrame that combines information from both the deliveries and matches Data
    """

    merged_df = deliveries_df.merge(
        matches_df,
        left_on='match_id',
        right_on='id',
        how='left'
    )

    return merged_df