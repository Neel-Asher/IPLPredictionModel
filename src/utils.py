def merge_datasets(deliveries_df, matches_df):

    merged_df = deliveries_df.merge(
        matches_df,
        left_on='match_id',
        right_on='id',
        how='left'
    )

    return merged_df