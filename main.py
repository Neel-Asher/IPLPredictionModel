from src.data_loader import load_deliveries_data, load_matches_data
from src.preprocessing import clean_column_names, clean_string_values
from src.utils import merge_datasets
from src.feature_engineering import *
from src.train_model import train_model, evaluate_model, save_model
from src.predict import *
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import matplotlib.pyplot as plt

DELIVERIES_PATH = "data/raw/deliveries.csv"
MATCHES_PATH = "data/raw/matches.csv"

def build_dataset():

    """
    Builds the dataset by loading, cleaning, and merging the deliveries and matches data.
    The function performs the following steps:
        1. Loads the deliveries and matches data from the specified paths.
        2. Cleans the column names of both DataFrames for consistency.
        3. Cleans string values in both DataFrames by stripping whitespace and replacing common placeholders for missing values with NaN.
        4. Merges the deliveries and matches DataFrames on the 'match_id' and 'id' columns, respectively, to create a single DataFrame that contains all relevant information for feature engineering and model training.
    Returns:
        pd.DataFrame: A merged and cleaned DataFrame ready for feature engineering and model training.  
    """

    deliveries_df = load_deliveries_data(DELIVERIES_PATH)
    matches_df = load_matches_data(MATCHES_PATH)

    deliveries_df = clean_column_names(deliveries_df)
    matches_df = clean_column_names(matches_df)

    deliveries_df = clean_string_values(deliveries_df)
    matches_df = clean_string_values(matches_df)

    merged_df = merge_datasets(deliveries_df, matches_df)

    merged_df = add_legal_delivery_column(merged_df)
    merged_df = add_dot_ball_column(merged_df)
    merged_df = add_boundary_ball_column(merged_df)

    merged_df["pitch_type"] = merged_df["venue"].apply(categorize_pitch)

    pitch_stats_df = create_pitch_type_stats(merged_df)
    adaptability_df = calculate_pitch_adaptability(pitch_stats_df)

    bowler_stats_df = create_bowler_season_stats(merged_df)
    bowler_stats_df = calculate_bowling_metrics(bowler_stats_df)

    bowler_stats_df = filter_bowlers_by_minimum_overs(
        bowler_stats_df,
        minimum_overs=20
    )

    bowler_stats_df = bowler_stats_df.merge(
        adaptability_df,
        on=["season", "bowler"],
        how="left"
    )

    bowler_stats_df = create_target_variable(bowler_stats_df)
    bowler_stats_df = bowler_stats_df.dropna(subset=["target_economy"])

    return bowler_stats_df

def train_pipeline(df):

    """
    Trains the machine learning model using the provided DataFrame by performing the following steps:
        1. Defines the feature columns and target variable for model training.
        2. Splits the data into training and testing sets using an 80-20 split.
        3. Trains a Random Forest Regressor model on the training data.
        4. Evaluates the model's performance on the test data by calculating key regression metrics (MAE, RMSE, R2) and printing the results.
        5. Saves the trained model to disk for future use.
    Parameters:
        df (pd.DataFrame): The DataFrame containing the data for training.  
    Returns:
        tuple: A tuple containing the trained model, the list of feature columns, and the original DataFrame used for training.
    """

    feature_columns = [
        "economy",
        "dot_ball_percentage",
        "boundary_ball_percentage",
        "bowling_strike_rate",
        "bowling_average",
        "pitch_adaptability_score",
        "matches_played"
    ]

    X = df[feature_columns]
    y = df["target_economy"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model)

    return model, feature_columns, df

def prediction_pipeline(df, model, feature_columns):

    """
    Predicts the economy for the next season using the trained model and identifies the top 5 bowlers with the lowest predicted economy rates. The function performs the following steps:
        1. Filters the input DataFrame to include only data from the latest season.
        2. Prepares the feature set for the latest season using the specified feature columns.
        3. Uses the trained model to predict the economy for the latest season.
        4. Adds the predicted economy values to the latest season DataFrame.
        5. Sorts the DataFrame by predicted economy in ascending order and selects the top 5 bowlers with the lowest predicted economy rates.
        6. Prints the top 5 bowlers and their predicted economy rates.
        7. Saves the top 5 predictions to a CSV file in the output directory.
        8. Creates a scatter plot of the top 5 bowlers and their predicted economy rates, highlighting the best predicted bowler and the average economy rate, and saves the plot to the output directory.
    Parameters:
        df (pd.DataFrame): The original DataFrame containing the data for prediction.
        model (object): The trained machine learning model to be used for prediction.
        feature_columns (list): A list of column names to be used as features for prediction.
    Returns:
        None
    """

    latest_season_df = df[
        df["season"] == df["season"].max()
    ].copy()

    X_latest = latest_season_df[feature_columns]

    predictions = model.predict(X_latest)

    latest_season_df["predicted_economy"] = predictions

    top_5 = latest_season_df.sort_values(
        by="predicted_economy",
        ascending=True
    ).head(5)

    top_5 = top_5.sort_values(
        by="predicted_economy"
    ).reset_index(drop=True)

    print("\nTop 5 Bowlers (Predicted Lowest Economy):")
    print(top_5[["bowler", "predicted_economy"]])

    output_dir = "output"

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "top_5_bowlers_prediction.csv"
    )

    top_5[
        ["bowler", "predicted_economy"]
    ].to_csv(
        output_path,
        index=False
    )

    print(f"\nSaved top 5 predictions to: {output_path}")


    plt.style.use("ggplot")

    plt.figure(figsize=(12, 7))

    colors = [
        "green",
        "limegreen",
        "orange",
        "darkorange",
        "red"
    ]

    # Plot all bowlers
    plt.scatter(
        top_5["bowler"],
        top_5["predicted_economy"],
        c=colors,
        s=220,
        edgecolors="black",
        linewidths=1.2
    )

    best_bowler = top_5.iloc[0]

    plt.scatter(
        best_bowler["bowler"],
        best_bowler["predicted_economy"],
        s=450,
        marker="*",
        color="gold",
        edgecolors="black",
        linewidths=1.5,
        label="Best Predicted Bowler"
    )

    for _, row in top_5.iterrows():

        plt.text(
            x=row["bowler"],
            y=row["predicted_economy"] + 0.04,
            s=f"{row['predicted_economy']:.2f}",
            ha="center",
            fontsize=10,
            fontweight="bold"
        )

    average_economy = top_5[
        "predicted_economy"
    ].mean()

    plt.axhline(
        y=average_economy,
        linestyle="--",
        linewidth=1.5,
        label=f"Average Economy ({average_economy:.2f})"
    )

    plt.xlabel(
        "Bowler",
        fontsize=12,
        fontweight="bold"
    )

    plt.ylabel(
        "Predicted Economy",
        fontsize=12,
        fontweight="bold"
    )

    plt.title(
        "Top 5 Bowlers Predicted for Next IPL Season",
        fontsize=15,
        fontweight="bold"
    )

    plt.suptitle(
        "Forecast Based on Historical IPL Bowling Data",
        fontsize=10
    )

    plt.xticks(
        rotation=15,
        fontsize=10
    )

    plt.yticks(fontsize=10)

    plt.grid(alpha=0.3)

    plt.legend()

    plt.tight_layout()

    plot_path = os.path.join(
        output_dir,
        "top_5_bowlers_scatter.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"\nScatter plot saved to: {plot_path}")

def main(): 

    """
    Main function to execute the entire pipeline for building the dataset, training the model, and making predictions for the next season. The function performs the following steps:
        1. Builds the dataset by loading, cleaning, and merging the deliveries and matches data
        2. Trains the machine learning model using the built dataset and evaluates its performance.
        3. Loads the trained model from disk and uses it to predict the economy for the next season.
        4. Identifies the top 5 bowlers with the lowest predicted economy rates and saves the results to a CSV file and a scatter plot in the output directory.
    Parameters:
        None
    Returns:
        None
    """

    df = build_dataset()
    model, feature_columns, df = train_pipeline(df)
    model = load_model()
    prediction_pipeline(df, model, feature_columns)

if __name__ == "__main__":
    main()