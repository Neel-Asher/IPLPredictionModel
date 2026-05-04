import pandas as pd

def load_deliveries_data(path):
    deliveries_df = pd.read_csv(path)
    return deliveries_df

def load_matches_data(path):
    matches_df = pd.read_csv(path)
    return matches_df