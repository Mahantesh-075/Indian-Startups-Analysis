import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from src.utils.config_loader import load_config, logger
from src.feature_engineering.engineer import run_feature_engineering

def train_success_classifier(df_funding, config, models_dir):
    """Trains a Random Forest Classifier to predict startup success probability."""
    logger.info("Training Startup Success Classifier...")
    df = df_funding.copy()
    
    # Define Target: "Success / High Growth" (1) if total funding is above median AND funding rounds >= 2
    median_funding = df["Total_Funding_USD"].median()
    df["Success_Target"] = ((df["Total_Funding_USD"] >= median_funding) | (df["Funding_Rounds"] >= 3)).astype(int)
    
    # Label encode categorical columns
    le_industry = LabelEncoder()
    le_city = LabelEncoder()
    
    df["Industry_Encoded"] = le_industry.fit_transform(df["Standard_Industry"])
    df["City_Encoded"] = le_city.fit_transform(df["City"])
    
    # Features & Target
    features = ["Industry_Encoded", "City_Encoded", "GDP_Growth", "Inflation", "FDI_Inflow"]
    X = df[features]
    y = df["Success_Target"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["models"]["test_size"], random_state=config["models"]["random_state"]
    )
    
    clf = RandomForestClassifier(
        n_estimators=config["models"]["n_estimators"], 
        random_state=config["models"]["random_state"]
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    train_preds = clf.predict(X_train)
    test_preds = clf.predict(X_test)
    train_acc = accuracy_score(y_train, train_preds)
    test_acc = accuracy_score(y_test, test_preds)
    
    logger.info(f"Classifier Metrics:\n  - Train Accuracy: {train_acc:.4f}\n  - Test Accuracy: {test_acc:.4f}")
    
    # Save model and encoders
    model_path = os.path.join(models_dir, config["files"]["success_model"])
    model_data = {
        "model": clf,
        "le_industry": le_industry,
        "le_city": le_city,
        "features": features
    }
    joblib.dump(model_data, model_path)
    logger.info(f"Saved success classifier to {model_path}")
    
    return train_acc, test_acc

def train_funding_regressor(df_funding, config, models_dir):
    """Trains a Random Forest Regressor to estimate startup funding amount."""
    logger.info("Training Startup Funding Regressor...")
    df = df_funding.copy()
    
    # Encoders
    le_industry = LabelEncoder()
    le_city = LabelEncoder()
    
    df["Industry_Encoded"] = le_industry.fit_transform(df["Standard_Industry"])
    df["City_Encoded"] = le_city.fit_transform(df["City"])
    
    # Features & Target
    features = ["Industry_Encoded", "City_Encoded", "GDP_Growth", "Inflation", "FDI_Inflow"]
    X = df[features]
    y = df["Funding_Amount_USD"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["models"]["test_size"], random_state=config["models"]["random_state"]
    )
    
    reg = RandomForestRegressor(
        n_estimators=config["models"]["n_estimators"],
        random_state=config["models"]["random_state"]
    )
    reg.fit(X_train, y_train)
    
    # Evaluate
    train_preds = reg.predict(X_train)
    test_preds = reg.predict(X_test)
    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)
    
    logger.info(f"Regressor Metrics:\n  - Train R2: {train_r2:.4f}\n  - Test R2: {test_r2:.4f}")
    
    # Save model and encoders
    model_path = os.path.join(models_dir, config["files"]["funding_model"])
    model_data = {
        "model": reg,
        "le_industry": le_industry,
        "le_city": le_city,
        "features": features
    }
    joblib.dump(model_data, model_path)
    logger.info(f"Saved funding regressor to {model_path}")
    
    return train_r2, test_r2

def train_macro_forecaster(df_gov, config, models_dir):
    """Fits trend models to forecast registered startup count growth up to 2030."""
    logger.info("Training Macro Startup Registration Forecaster...")
    
    # Get total registrations by year
    yearly_registrations = df_gov.groupby("Year")["Count"].sum().reset_index()
    yearly_registrations = yearly_registrations[yearly_registrations["Year"] <= 2025]
    
    X = yearly_registrations["Year"].values.reshape(-1, 1)
    y = yearly_registrations["Count"].values
    
    # Fit quadratic polynomial model to capture the hockey stick growth curve
    poly_features = np.hstack([X, X**2])
    
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(poly_features, y)
    
    r2 = r2_score(y, reg.predict(poly_features))
    logger.info(f"Macro Forecaster R2 score: {r2:.4f}")
    
    # Save model weights
    model_path = os.path.join(models_dir, config["files"]["forecast_model"])
    model_data = {
        "coefficients": reg.coef_,
        "intercept": reg.intercept_,
        "r2": r2
    }
    joblib.dump(model_data, model_path)
    logger.info(f"Saved macro forecaster weights to {model_path}")
    
    return r2

def run_model_training():
    """Main execution function to train all ML models and save weight files."""
    config = load_config()
    proc_funding, proc_gov = run_feature_engineering()
    
    models_dir = config["resolved_paths"]["models_dir"]
    
    train_success_classifier(proc_funding, config, models_dir)
    train_funding_regressor(proc_funding, config, models_dir)
    train_macro_forecaster(proc_gov, config, models_dir)
    
    logger.info("All model training and serialization completed successfully!")

if __name__ == "__main__":
    run_model_training()
