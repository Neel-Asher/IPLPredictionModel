import numpy as np

def add_legal_delivery_column(df):

    """
    Adds a column 'is_legal_delivery' to the DataFrame, where:
        - 1 indicates a legal delivery (not a wide or no-ball)
        - 0 indicates an illegal delivery (wide or no-ball)
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the 'extras_type' column.
    Returns:
        pd.DataFrame: The DataFrame with the new 'is_legal_delivery' column added.
    """

    df['is_legal_delivery'] = np.where(
        df['extras_type'].isin(['wides', 'noballs']),
        0,
        1
    )

    return df

def add_dot_ball_column(df):

    """
    Adds a column 'is_dot_ball' to the DataFrame, where:
        - 1 indicates a dot ball (no runs scored and is a legal delivery)
        - 0 indicates a non-dot ball
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the relevant columns.
    Returns:
        pd.DataFrame: The DataFrame with the new 'is_dot_ball' column added.
    """

    df['is_dot_ball'] = (
        (df['total_runs'] == 0) &
        (df['is_legal_delivery'] == 1)
    ).astype(int)

    return df

def add_boundary_ball_column(df):

    """
    Adds a column 'is_boundary' to the DataFrame, where:
        - 1 indicates a boundary ball (4 or 6 runs scored)
        - 0 indicates a non-boundary ball
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the 'batsman_runs' column.
    Returns:
        pd.DataFrame: The DataFrame with the new 'is_boundary' column added.
    """

    df['is_boundary'] = (
        df['batsman_runs'].isin([4, 6])
    ).astype(int)

    return df

def create_bowler_season_stats(df):

    """
    Aggregates the deliveries DataFrame to create season-level statistics for each bowler.
    The resulting DataFrame contains the following columns:
        - 'season': The season of the IPL.
        - 'bowler': The name of the bowler.
        - 'total_balls': The total number of legal deliveries bowled by the bowler in that season.
        - 'total_runs_conceded': The total number of runs conceded by the bowler in that season.
    """

    grouped_df = df.groupby(
        ['season', 'bowler']
    ).agg(
        total_balls=('is_legal_delivery', 'sum'),
        total_runs_conceded=('total_runs', 'sum'),
        dot_balls=('is_dot_ball', 'sum'),
        boundary_balls=('is_boundary', 'sum'),
        wickets=('is_wicket', 'sum'),
        matches_played=('match_id', 'nunique')
    ).reset_index()

    return grouped_df

def calculate_bowling_metrics(df):

    """
    Calculates key bowling metrics for each bowler-season combination, including:
        - 'overs_bowled': Total overs bowled (total_balls / 6
        - 'economy': Runs conceded per over (total_runs_conceded / overs_bowled)
        - 'dot_ball_percentage': Percentage of deliveries that were dot balls (dot_balls / total_balls * 100)
        - 'boundary_ball_percentage': Percentage of deliveries that were boundary balls (boundary_balls / total_balls * 100)
    """

    df['overs_bowled'] = df['total_balls'] / 6

    df['economy'] = (
        df['total_runs_conceded'] /
        df['overs_bowled']
    )

    df['dot_ball_percentage'] = (
        df['dot_balls'] /
        df['total_balls']
    ) * 100

    df['boundary_ball_percentage'] = (
        df['boundary_balls'] /
        df['total_balls']
    ) * 100

    df['bowling_strike_rate'] = (
        df['total_balls'] /
        df['wickets'].replace(0, 1)
    )

    df['bowling_average'] = (
        df['total_runs_conceded'] /
        df['wickets'].replace(0, 1)
    )

    return df

def filter_bowlers_by_minimum_overs(df,minimum_overs=20):

    """
    Filters the DataFrame to include only bowlers who have bowled a minimum number of overs in a season.
    Parameters:
        df (pd.DataFrame): The input DataFrame containing the 'overs_bowled' column.
        minimum_overs (int): The minimum number of overs a bowler must have bowled in a season to be included in the output DataFrame.
    Returns:    
        pd.DataFrame: A filtered DataFrame containing only bowlers who have bowled at least the specified number of overs.
    """

    filtered_df = df[
        df['overs_bowled'] >= minimum_overs
    ]

    return filtered_df

def categorize_pitch(venue):

    """
    Categorizes the pitch type based on the venue. The function classifies venues into three categories:
        - 'batting': Venues known for being batting-friendly.   
        - 'spin': Venues known for being spin-friendly.
        - 'pace': Venues known for being pace-friendly.
        - 'balanced': Venues that do not strongly favor either batsmen or bowlers.
    Parameters:
        venue (str): The name of the venue where the match is played.
    Returns:
        str: The category of the pitch ('batting', 'spin', 'pace', or 'balanced').
    """

    batting_friendly = [
        'M Chinnaswamy Stadium',
        'Wankhede Stadium',
        'Brabourne Stadium',
        'Arun Jaitley Stadium',
        'Rajiv Gandhi International Stadium'
    ]

    spin_friendly = [
        'MA Chidambaram Stadium',
        'Eden Gardens'
    ]

    pace_friendly = [
        'Punjab Cricket Association Stadium',
        'Punjab Cricket Association IS Bindra Stadium',
        'Himachal Pradesh Cricket Association Stadium'
    ]

    if venue in batting_friendly:
        return 'batting'

    elif venue in spin_friendly:
        return 'spin'

    elif venue in pace_friendly:
        return 'pace'

    else:
        return 'balanced'
    
def create_pitch_type_stats(df):

    """
    Creates a DataFrame containing statistics for each bowler-season-pitch_type combination, including:
        - 'season': The season of the IPL.
        - 'bowler': The name of the bowler.
        - 'pitch_type': The categorized pitch type for the match.
        - 'total_balls': The total number of legal deliveries bowled by the bowler on that pitch type in that season.
        - 'total_runs': The total number of runs conceded by the bowler on that pitch type in that season.
        - 'overs': The total number of overs bowled by the bowler on that pitch type in that season (total_balls / 6).
    """

    pitch_stats_df = df.groupby(
        ['season', 'bowler', 'pitch_type']
    ).agg(
        total_balls=('is_legal_delivery', 'sum'),
        total_runs=('total_runs', 'sum')
    ).reset_index()

    pitch_stats_df['overs'] = (
        pitch_stats_df['total_balls'] / 6
    )

    pitch_stats_df['pitch_economy'] = (
        pitch_stats_df['total_runs'] /
        pitch_stats_df['overs']
    )

    return pitch_stats_df

def calculate_pitch_adaptability(pitch_stats_df):

    """
    Calculates a pitch adaptability score for each bowler-season combination based on the variance of their economy rates across different pitch types. The function performs the following steps:
        1. Groups the pitch_stats_df by 'season' and 'bowler' and calculates the variance of 'pitch_economy' for each group.
        2. Fills any NaN values in the variance column with 0 (indicating no variability).
        3. Calculates the pitch adaptability score as the inverse of (variance + 1) to ensure that higher variability results in a lower adaptability score.    
    """

    adaptability_df = pitch_stats_df.groupby(
        ['season', 'bowler']
    ).agg(
        pitch_economy_variance=(
            'pitch_economy',
            'var'
        )
    ).reset_index() 

    adaptability_df[
        'pitch_economy_variance'
    ] = adaptability_df[
        'pitch_economy_variance'
    ].fillna(0)

    adaptability_df[
        'pitch_adaptability_score'
    ] = 1 / (
        adaptability_df[
            'pitch_economy_variance'
        ] + 1
    )

    return adaptability_df

def create_target_variable(df):

    """
    Creates the target variable 'target_economy' for each bowler-season combination by shifting the 'economy' column upwards by one season. This means that the 'target_economy' for a given season will represent the economy rate of the next season for that bowler. The function performs the following steps:
        1. Sorts the DataFrame by 'bowler' and 'season' to ensure that the data is in the correct order for shifting.
        2. Creates the 'target_economy' column by grouping the DataFrame by 'bowler' and applying the shift function to the 'economy' column with a shift of -1 (upwards).      
    """

    df = df.sort_values(
        by=['bowler', 'season']
    )

    df['target_economy'] = (
        df.groupby('bowler')['economy']
        .shift(-1)
    )

    return df