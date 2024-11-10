import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
import warnings

from helpers import *
from config import classifiers

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

# Veri hazırlama fonksiyonu (ev fiyatı veri seti)
def house_price_data_prep(dataframe):
    # Genel veri kontrolü
    check_df(dataframe)

    # Kategorik ve sayısal değişkenleri yakalama
    cat_cols, cat_but_car, num_cols = grab_col_names(dataframe)

    # Aykırı değerler ile başa çıkma
    for col in num_cols:
        if col != "SalePrice":
            replace_with_thresholds(dataframe, col)

    # Fill missing values for specific columns
    no_cols = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu",
               "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]

    for col in no_cols:
        if col in dataframe.columns:  # Check if the column exists in the DataFrame
            dataframe[col] = dataframe[col].fillna("No")

    missing_values_table(dataframe)
    dataframe = quick_missing_imp(dataframe, num_method="median", cat_length=17)

    # Rare encoding
    dataframe = rare_encoder(dataframe, 0.01)

    # Özellik mühendisliği (feature engineering)
    dataframe["NEW_1st*GrLiv"] = dataframe["1stFlrSF"] * dataframe["GrLivArea"]
    dataframe["NEW_Garage*GrLiv"] = dataframe["GarageArea"] * dataframe["GrLivArea"]
    dataframe["NEW_TotalFlrSF"] = dataframe["1stFlrSF"] + dataframe["2ndFlrSF"]
    dataframe["NEW_TotalBsmtFin"] = dataframe.BsmtFinSF1 + dataframe.BsmtFinSF2
    dataframe["NEW_PorchArea"] = dataframe.OpenPorchSF + dataframe.EnclosedPorch + dataframe.ScreenPorch + dataframe[
        "3SsnPorch"] + dataframe.WoodDeckSF
    dataframe["NEW_TotalHouseArea"] = dataframe.NEW_TotalFlrSF + dataframe.TotalBsmtSF
    dataframe["NEW_TotalSqFeet"] = dataframe.GrLivArea + dataframe.TotalBsmtSF
    dataframe["NEW_LotRatio"] = dataframe.GrLivArea / dataframe.LotArea
    dataframe["NEW_RatioArea"] = dataframe.NEW_TotalHouseArea / dataframe.LotArea
    dataframe["NEW_GarageLotRatio"] = dataframe.GarageArea / dataframe.LotArea
    dataframe["NEW_MasVnrRatio"] = dataframe.MasVnrArea / dataframe.NEW_TotalHouseArea
    dataframe["NEW_DifArea"] = (dataframe.LotArea - dataframe["1stFlrSF"] - dataframe.GarageArea - dataframe.NEW_PorchArea - dataframe.WoodDeckSF)
    dataframe["NEW_OverallGrade"] = dataframe["OverallQual"] * dataframe["OverallCond"]
    dataframe["NEW_Restoration"] = dataframe.YearRemodAdd - dataframe.YearBuilt
    dataframe["NEW_HouseAge"] = dataframe.YrSold - dataframe.YearBuilt
    dataframe["NEW_RestorationAge"] = dataframe.YrSold - dataframe.YearRemodAdd
    dataframe["NEW_GarageAge"] = dataframe.GarageYrBlt - dataframe.YearBuilt
    dataframe["NEW_GarageRestorationAge"] = abs(dataframe.GarageYrBlt - dataframe.YearRemodAdd)
    dataframe["NEW_GarageSold"] = dataframe.YrSold - dataframe.GarageYrBlt

    # Gereksiz sütunların silinmesi
    drop_list = ["Street", "Alley", "LandContour", "Utilities", "LandSlope", "Heating", "PoolQC", "MiscFeature",
                 "Neighborhood"]
    dataframe.drop(drop_list, axis=1, inplace=True)

    # Binary encoding for categorical variables
    binary_cols = [col for col in dataframe.columns if
                   dataframe[col].dtype == "O" and len(dataframe[col].unique()) == 2]
    for col in binary_cols:
        dataframe = label_encoder(dataframe, col)

    # One-hot encoding for categorical variables
    existing_cols = [col for col in cat_cols if col in dataframe.columns]
    dataframe = one_hot_encoder(dataframe, existing_cols, drop_first=True)

    X = dataframe.drop(["Id", "SalePrice"], axis=1)
    y = dataframe["SalePrice"]

    return X, y



# Temel modellerin performanslarını kontrol etme
def base_models(X, y):
    models = [('LR', LinearRegression()),
              ('CART', DecisionTreeRegressor()),
              ('RF', RandomForestRegressor()),
              ('GBM', GradientBoostingRegressor()),
              ("XGBoost", XGBRegressor(objective='reg:squarederror')),
              ("LightGBM", LGBMRegressor()),
              ("CatBoost", CatBoostRegressor(verbose=False))]

    print("Normal Model Performansları:")
    for name, regressor in models:
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
        print(f"RMSE: {round(rmse, 4)} ({name})")

    return models


# Hyperparametre optimizasyonu
def hyperparameter_optimization(X, y, cv=5, scoring="neg_mean_squared_error"):
    print("Hyperparameter Optimization....")
    best_models = {}

    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")

        # Hyperparameter optimization using GridSearchCV
        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, scoring=scoring, verbose=False).fit(X, y)

        # En iyi parametrelerle model oluşturma
        final_model = classifier.set_params(**gs_best.best_params_)

        best_models[name] = final_model

    return best_models


# Voting Regressor modeli
def voting_regressor(best_models, X, y):
    print("Voting Regressor...")

    # Voting Regressor modelini tanımlama
    voting_reg = VotingRegressor(estimators=[
        ('XGBoost', best_models["XGBoost"]),
        ('LightGBM', best_models["LightGBM"]),
        ('CatBoost', best_models["CatBoost"])
    ]).fit(X, y)

    return voting_reg


################################################
# Pipeline Main Function
################################################

def main():
    # Read the data
    train = pd.read_csv("datasets/train.csv")
    test = pd.read_csv("datasets/test.csv")

    # Combine training and test sets
    df = pd.concat([train, test], ignore_index=True)

    # Data preparation
    X, y = house_price_data_prep(train)  # Prepare the training data

    # Split the training data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=41)

    # Check performance of base models
    base_models(X_train, y_train)

    # Perform hyperparameter optimization
    best_models = hyperparameter_optimization(X_train, y_train)

    # Create Voting Regressor model
    final_model = voting_regressor(best_models, X_train, y_train)

    # Save the model
    joblib.dump(final_model, "voting_regressor_model.pkl")

    # Check performance on validation set
    val_predictions = final_model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
    print(f"Validation RMSE: {round(val_rmse, 4)}")

    # Prepare test data and make predictions
    test_data_prep, _ = house_price_data_prep(test)  # Prepare test data (ignore y as it doesn't exist)
    predictions = final_model.predict(test_data_prep)  # Make predictions

    # Create DataFrame for Kaggle submission
    submission = pd.DataFrame({"Id": test["Id"], "SalePrice": predictions})  # DataFrame with Id and predictions
    submission.to_csv("housePricePredictions1.csv", index=False)  # Write to CSV file

if __name__ == "__main__":
    main()