import joblib
import pandas as pd

def load_model(path="models/random_forest_model.pkl"):

    model = joblib.load(path)
    return model

def predict_next_season_economy(model, X):

    predictions = model.predict(X)
    return predictions

def get_top_bowlers(bowler_stats_df, predictions, top_n=5):

    result_df = bowler_stats_df.copy()
    result_df["predicted_economy"] = predictions

    top_bowlers = result_df.sort_values(
        by="predicted_economy",
        ascending=True
    ).head(top_n)

    return top_bowlers[
        [
            "season",
            "bowler",
            "predicted_economy"
        ]
    ]