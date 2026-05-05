import joblib
import pandas as pd

def load_model(path="models/random_forest_model.pkl"):

    """
    Loads a trained model from the specified path.
    Parameters:
        path (str): The path to the saved model file.
    Returns:
        object: The loaded model.
    """

    model = joblib.load(path)
    return model

def predict_next_season_economy(model, X):

    """
    Predicts the economy for the next season using the trained model.
    Parameters:
        model (object): The trained model.
        X (pd.DataFrame): The input features for prediction.
    Returns:
        np.ndarray: The predicted economy values.
    """

    predictions = model.predict(X)
    return predictions

def get_top_bowlers(bowler_stats_df, predictions, top_n=5):

    """
    Gets the top bowlers with the lowest predicted economy rates based on the provided predictions and bowler statistics DataFrame. The function performs the following steps:
        1. Creates a copy of the bowler_stats_df to avoid modifying the original DataFrame.
        2. Adds a new column 'predicted_economy' to the copied DataFrame, containing the predicted economy values.
        3. Sorts the DataFrame by 'predicted_economy' in ascending order to get bowlers with the lowest predicted economy rates at the top.
        4. Selects the top N bowlers based on the sorted order.
        5. Returns a DataFrame containing the season, bowler name, and predicted economy for the top N bowlers.
    """

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