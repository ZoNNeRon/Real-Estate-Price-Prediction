"""
feature_engineering.py - Feature creation and transformation functions

This module handles:
1. Creating new features from existing ones
2. Encoding categorical variables
3. Scaling numerical features
4. Feature selection
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression


def engineer_features(df, target_col=None, is_training=True):
    """
    Create and engineer features for the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column (if exists)
    is_training : bool
        If True, create encoders; if False, use existing encoders
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with engineered features
    """
    df = df.copy()
    
    # Identify dataset type (Ames vs Malaysia)
    if 'SalePrice' in df.columns or 'Id' in df.columns and df.shape[1] > 50:
        return engineer_ames_features(df, target_col, is_training)
    else:
        return engineer_malaysia_features(df, target_col, is_training)


def engineer_ames_features(df, target_col='SalePrice', is_training=True):
    """
    Engineer features specific to Ames dataset.
    
    Process:
    1. Handle missing values with domain knowledge
    2. Create interaction features
    3. Encode categorical variables
    4. Scale numerical features
    """
    df = df.copy()
    
    # STEP 1: Handle missing values with domain knowledge
    # Columns with 'NaN' meaning "no feature" (not missing data)
    fill_none_cols = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                      'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 
                      'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
    for col in fill_none_cols:
        if col in df.columns:
            df[col] = df[col].fillna('None')
    
    # Fill numerical missing values
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # STEP 2: Create interaction features
    if 'GrLivArea' in df.columns and 'LotArea' in df.columns:
        df['LivingArea_to_LotArea'] = df['GrLivArea'] / (df['LotArea'] + 1)
    
    if 'TotalBsmtSF' in df.columns and 'GrLivArea' in df.columns:
        df['TotalArea'] = df['TotalBsmtSF'] + df['GrLivArea']
    
    if 'GarageCars' in df.columns and 'GarageArea' in df.columns:
        df['GarageArea_per_Car'] = df['GarageArea'] / (df['GarageCars'] + 1)
    
    # STEP 3: Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df_encoded


def engineer_malaysia_features(df, target_col='Median_Price', is_training=True):
    """
    Engineer features specific to Malaysia dataset.
    
    Process:
    1. Handle missing values
    2. Normalize price and PSF
    3. Create location interaction features
    4. Scale features
    """
    df = df.copy()
    
    # STEP 1: Handle missing values
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # STEP 2: Create features from existing data
    if 'Median_Price' in df.columns:
        df['LogPrice'] = np.log1p(df['Median_Price'])
    
    if 'MedianPSF' in df.columns:
        df['LogPSF'] = np.log1p(df['MedianPSF'])
    
    # STEP 3: Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df_encoded


def select_best_features(X, y, k=20):
    """
    Select top k features based on f-regression score.
    
    Parameters:
    -----------
    X : pd.DataFrame or np.ndarray
        Feature matrix
    y : pd.Series or np.ndarray
        Target variable
    k : int
        Number of features to select
        
    Returns:
    --------
    list
        Names of selected features
    """
    selector = SelectKBest(f_regression, k=k)
    selector.fit(X, y)
    
    if isinstance(X, pd.DataFrame):
        selected_features = X.columns[selector.get_support()].tolist()
    else:
        selected_features = list(range(X.shape[1]))
        selected_features = [f for i, f in enumerate(selected_features) if selector.get_support()[i]]
    
    return selected_features


def scale_features(X_train, X_test=None):
    """
    Standardize features using StandardScaler.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame, optional
        Test features to scale using training scaler
        
    Returns:
    --------
    tuple or pd.DataFrame
        Scaled training (and test) features
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler