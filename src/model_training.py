from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

class ModelTrainer:
    """Train and evaluate regression models."""
    
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.scaler = StandardScaler()
        self.models = {}
    
    def scale_data(self):
        """Standardize features."""
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
    
    def train_lasso(self, alpha=1.0):
        """Train Lasso Regression model."""
        model = Lasso(alpha=alpha, random_state=42)
        model.fit(self.X_train_scaled, self.y_train)
        self.models['lasso'] = model
        return model
    
    def train_random_forest(self, n_estimators=100, max_depth=15):
        """Train Random Forest model."""
        model = RandomForestRegressor(n_estimators=n_estimators, 
                                     max_depth=max_depth, 
                                     random_state=42)
        model.fit(self.X_train, self.y_train)
        self.models['random_forest'] = model
        return model
    
    def train_xgboost(self, max_depth=6, learning_rate=0.1, n_estimators=100):
        """Train XGBoost model."""
        model = xgb.XGBRegressor(max_depth=max_depth, 
                                learning_rate=learning_rate,
                                n_estimators=n_estimators, 
                                random_state=42)
        model.fit(self.X_train, self.y_train)
        self.models['xgboost'] = model
        return model
    
    def evaluate_model(self, model_name, use_scaled=False):
        """Evaluate model performance."""
        model = self.models[model_name]
        X_test = self.X_test_scaled if use_scaled else self.X_test
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'r2': r2_score(self.y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred)),
            'mae': mean_absolute_error(self.y_test, y_pred),
            'model': model_name
        }
        
        return metrics, y_pred
    
    def get_feature_importance(self, model_name, feature_names):
        """Extract feature importance scores."""
        model = self.models[model_name]
        
        if model_name == 'lasso':
            importance = np.abs(model.coef_)
        else:  # Random Forest and XGBoost
            importance = model.feature_importances_
        
        # Sort by importance
        indices = np.argsort(importance)[::-1]
        
        return {
            'features': [feature_names[i] for i in indices],
            'importance': importance[indices]
        }