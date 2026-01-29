"""
__init__.py - Package initialization file

This file makes the src directory a Python package, allowing imports like:
    from src.data_loader import load_data
    from src.feature_engineering import engineer_features
    etc.
"""

__version__ = "1.0.0"
__author__ = "Dmitrii Vasilov"
__description__ = "Real Estate Price Prediction Models"

# Import main functions for easier access
from .data_loader import load_data, preprocess_data
from .feature_engineering import engineer_features
from .model_training import train_model
from .model_evaluation import evaluate_model
from .visualization import plot_feature_importance

__all__ = [
    'load_data',
    'preprocess_data',
    'engineer_features',
    'train_model',
    'evaluate_model',
    'plot_feature_importance'
]