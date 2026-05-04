import numpy as np

def add_legal_delivery_column(df):

    df['is_legal_delivery'] = np.where(
        df['extras_type'].isin(['wides', 'noballs']),
        0,
        1
    )

    return df

def add_dot_ball_column(df):

    df['is_dot_ball'] = (
        (df['total_runs'] == 0) &
        (df['is_legal_delivery'] == 1)
    ).astype(int)

    return df

def add_boundary_ball_column(df):

    df['is_boundary'] = (
        df['batsman_runs'].isin([4, 6])
    ).astype(int)

    return df

def create_bowler_season_stats(df):

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

    filtered_df = df[
        df['overs_bowled'] >= minimum_overs
    ]

    return filtered_df

def categorize_pitch(venue):

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

    df = df.sort_values(
        by=['bowler', 'season']
    )

    df['target_economy'] = (
        df.groupby('bowler')['economy']
        .shift(-1)
    )

    return df