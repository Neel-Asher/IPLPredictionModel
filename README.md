# IPLPredictionModel
An IPL analytics project that uses Random Forest Regression to predict the top 5 bowlers with the lowest economy rates in the next season. The model leverages historical performance, pitch adaptability, match conditions, consistency, and advanced bowling metrics to generate data-driven predictions.


## Project Overview

This project builds a machine learning pipeline to predict bowling performance in future IPL seasons using historical match data. The primary objective is to forecast the expected economy rate of bowlers in the next season and identify the top performers (lowest predicted economy rates).

The model is trained on ball-by-ball and match-level IPL data spanning multiple seasons and uses engineered performance and contextual features to make predictions.

---

## Problem Statement

Given historical IPL data for bowlers across multiple seasons, the goal is to:

- Predict each bowler’s economy rate for the next season
- Rank bowlers based on predicted performance
- Identify the top 5 bowlers expected to have the lowest economy rates in the upcoming season

This is formulated as a supervised regression problem with a temporal target shift.

---

## Dataset Description

The project uses two primary datasets:

- `deliveries.csv`: Ball-by-ball match data including runs, wickets, extras, and bowler information  
- `matches.csv`: Match-level metadata including season, venue, and match identifiers  

---

## Feature Engineering

The project constructs multiple levels of features:

### Ball-level features
- Legal delivery indicator
- Dot ball indicator
- Boundary ball indicator

### Season-level aggregated features
- Total runs conceded
- Total balls bowled
- Economy rate
- Bowling strike rate
- Bowling average
- Dot ball percentage
- Boundary ball percentage
- Matches played

### Contextual features
- Pitch type classification (batting, spin, pace, balanced)
- Pitch adaptability score based on performance variance across pitch types

---

## Target Variable

The model predicts:

- **Target economy rate = bowler’s economy rate in the next season**

This is implemented using a temporal shift per bowler.

---

## Machine Learning Approach

- Model: Random Forest Regressor  
- Training Strategy: Supervised regression with time-shifted target  
- Evaluation Metrics:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score  

---

## Pipeline Workflow

1. Load raw datasets  
2. Clean and standardize data  
3. Merge deliveries and match information  
4. Engineer ball-level and season-level features  
5. Create target variable using next-season shift  
6. Split data into training and testing sets  
7. Train Random Forest regression model  
8. Evaluate model performance  
9. Save trained model  
10. Predict next-season economy rates  
11. Rank bowlers and output top performers  

---

## How to Run the Project

### 1. Install dependencies: pip install -r requirements.txt
### 2. Run the pipeline: python main.py


---

## Output

The system outputs:

- Model evaluation metrics (MAE, RMSE, R²)
- Top 5 bowlers predicted to have the lowest economy in the next season
- Saved trained model in `models/` directory

---

## Key Insights

- The model captures player consistency over multiple seasons  
- Pitch adaptability improves prediction stability  
- Economy rate is influenced by both performance and contextual match conditions  
- Predictions are based on historical statistical patterns  

---

## Limitations

- Assumes historical performance trends continue into future seasons  
- Does not account for injuries, team changes, or role changes  
- New players with limited history may be underrepresented  

---

## Future Improvements

- Hyperparameter tuning using GridSearchCV  
- Cross-season backtesting framework  
- Feature importance visualization  
- API deployment using FastAPI  
- Model versioning and experiment tracking  

---

## Author Notes

This project demonstrates an end-to-end machine learning pipeline covering:

- Data engineering  
- Feature engineering  
- Supervised learning for time-series style prediction  
- Sports analytics use case (IPL bowling performance forecasting)
