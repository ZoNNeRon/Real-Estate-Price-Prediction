import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataLoader:
    """Load and preprocess housing datasets."""
    
    def __init__(self, ames_path, malaysia_path):
        self.ames_df = None
        self.malaysia_df = None
        self.ames_path = ames_path
        self.malaysia_path = malaysia_path
    
    def load_ames_data(self):
        """Load Ames Housing dataset."""
        self.ames_df = pd.read_csv(self.ames_path)
        print(f"Ames data loaded: {self.ames_df.shape}")
        return self.ames_df
    
    def load_malaysia_data(self):
        """Load Malaysia Housing dataset."""
        self.malaysia_df = pd.read_csv(self.malaysia_path)
        print(f"Malaysia data loaded: {self.malaysia_df.shape}")
        return self.malaysia_df
    
    def handle_missing_values(self, df, threshold=50):
        """Remove features with >threshold% missing values."""
        missing_pct = (df.isnull().sum() / len(df)) * 100
        cols_to_drop = missing_pct[missing_pct > threshold].index
        df = df.drop(columns=cols_to_drop)
        
        # Impute remaining missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc)
        
        return df
    
    def train_test_split_data(self, df, target_col, test_size=0.2, random_state=42):
        """Split data into train/validation sets."""
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        return train_test_split(X, y, test_size=test_size, 
                               random_state=random_state)