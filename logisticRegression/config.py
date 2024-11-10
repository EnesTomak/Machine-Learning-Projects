from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Hyperparameter grids for XGBoost, LightGBM and CatBoost
xgboost_params = {
    "learning_rate": [0.1, 0.01],
    "max_depth": [5, 8],
    "n_estimators": [100, 200],
    "colsample_bytree": [0.5, 1]
}

lightgbm_params = {
    "learning_rate": [0.01, 0.1],
    "n_estimators": [300, 500],
    "colsample_bytree": [0.7, 1]
}

catboost_params = {
    "iterations": [200, 500],
    "learning_rate": [0.01, 0.1],
    "depth": [6, 8, 10],
    "l2_leaf_reg": [3, 5, 7],
}

# Classifiers list with corresponding hyperparameters
classifiers = [
    ('XGBoost', XGBRegressor(), xgboost_params),
    ('LightGBM', LGBMRegressor(), lightgbm_params),
    ('CatBoost', CatBoostRegressor(), catboost_params)
]
