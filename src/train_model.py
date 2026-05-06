import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_model(X_train, y_train):

    """
    Trains a Random Forest Regressor model using the provided training data (X_train and y_train). The function initializes the model with specified hyperparameters, 
    fits the model to the training data, and returns the trained model. The hyperparameters used in this implementation include:
        - n_estimators: The number of trees in the forest (set to 200). 
        - random_state: A seed value for reproducibility (set to 42).
        - n_jobs: The number of jobs to run in parallel for both fit and predict (set to -1 to use all available processors).
    Parameters:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target values.
    Returns:
        RandomForestRegressor: The trained Random Forest Regressor model.
    """

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):

    """
    Evaluates the performance of the trained model on the test data (X_test and y_test) by calculating key regression metrics, including Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R2). The function generates predictions using the model, computes the evaluation metrics, and prints the results in a readable format.
    Parameters:
        model (RandomForestRegressor): The trained Random Forest Regressor model to be evaluated
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The true target values for the test set.
    Returns:
        tuple: A tuple containing the MAE, RMSE, and R2 scores.
    """

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print("\nModel Evaluation Metrics:")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    return mae, rmse, r2

def save_model(model, path="models/random_forest_model.pkl"):

    """
    Saves the trained model to the specified path.
    Parameters:
        model (RandomForestRegressor): The trained Random Forest Regressor model to be saved.
        path (str): The file path where the model will be saved.
    """
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"\nModel saved at: {path}")