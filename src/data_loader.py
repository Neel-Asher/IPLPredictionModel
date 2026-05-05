import pandas as pd

def load_deliveries_data(path):

    """
    Loads the deliveries data from the specified path.
    Parameters:
        path (str): The path to the deliveries CSV file.
    Returns:
        pd.DataFrame: A DataFrame containing the deliveries data.
    """

    deliveries_df = pd.read_csv(path)
    return deliveries_df

def load_matches_data(path):

    """
    Loads the matches data from the specified path.
    Parameters:
        path (str): The path to the matches CSV file.
    Returns:
        pd.DataFrame: A DataFrame containing the matches data.
    """

    matches_df = pd.read_csv(path)
    return matches_df