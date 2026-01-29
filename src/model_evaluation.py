"""
model_evaluation.py - Model evaluation metrics and analysis

This module provides:
1. Performance metrics (R², RMSE, MAE, MAPE)
2. Cross-validation evaluation
3. Residual analysis
4. Model comparison functions
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


def calculate_metrics(y_true, y_pred, model_name="Model"):
    """
    Calculate comprehensive evaluation metrics.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    model_name : str
        Name of the model for reporting
        
    Returns:
    --------
    dict
        Dictionary with all metrics
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Calculate relative error
    mean_price = np.mean(y_true)
    relative_error = (rmse / mean_price) * 100
    
    metrics = {
        'Model': model_name,
        'R2_Score': r2,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Relative_Error_Percent': relative_error
    }
    
    return metrics


def print_metrics(metrics):
    """
    Pretty print model metrics.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary with metrics from calculate_metrics
    """
    print(f"\n{'='*50}")
    print(f"Model: {metrics['Model']}")
    print(f"{'='*50}")
    print(f"R² Score:              {metrics['R2_Score']:.4f} (higher is better)")
    print(f"RMSE:                  ${metrics['RMSE']:.2f}" if metrics['RMSE'] > 1000 else f"{metrics['RMSE']:.2f}")
    print(f"MAE:                   ${metrics['MAE']:.2f}" if metrics['MAE'] > 1000 else f"{metrics['MAE']:.2f}")
    print(f"MAPE:                  {metrics['MAPE']:.2f}%")
    print(f"Relative Error:        ±{metrics['Relative_Error_Percent']:.2f}%")
    print(f"{'='*50}\n")


def compare_models(predictions_dict, y_true):
    """
    Compare multiple model predictions.
    
    Parameters:
    -----------
    predictions_dict : dict
        Dictionary {model_name: predictions}
    y_true : array-like
        Actual values
        
    Returns:
    --------
    pd.DataFrame
        Comparison table with all metrics
    """
    results = []
    
    for model_name, y_pred in predictions_dict.items():
        metrics = calculate_metrics(y_true, y_pred, model_name)
        results.append(metrics)
    
    comparison_df = pd.DataFrame(results).sort_values('R2_Score', ascending=False)
    return comparison_df


def plot_predictions(y_true, y_pred, model_name="Model", title="Actual vs Predicted"):
    """
    Plot actual vs predicted values with residuals.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    model_name : str
        Name of the model
    title : str
        Plot title
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Actual vs Predicted
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Price')
    axes[0, 0].set_ylabel('Predicted Price')
    axes[0, 0].set_title(f'{model_name} - Actual vs Predicted')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Price')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Residual Distribution
    axes[1, 0].hist(residuals, bins=30, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Residual Distribution')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Error Distribution
    errors = np.abs(residuals)
    axes[1, 1].hist(errors, bins=30, edgecolor='black', color='orange')
    axes[1, 1].set_xlabel('Absolute Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Absolute Error Distribution')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def cross_validate_model(model, X, y, cv=5):
    """
    Perform k-fold cross-validation.
    
    Parameters:
    -----------
    model : sklearn estimator
        ML model with fit and predict methods
    X : array-like
        Feature matrix
    y : array-like
        Target variable
    cv : int
        Number of folds
        
    Returns:
    --------
    dict
        Cross-validation metrics
    """
    from sklearn.model_selection import cross_val_score
    
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    rmse_scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(rmse_scores)
    
    cv_results = {
        'R2_Mean': r2_scores.mean(),
        'R2_Std': r2_scores.std(),
        'RMSE_Mean': rmse_scores.mean(),
        'RMSE_Std': rmse_scores.std(),
        'R2_Scores': r2_scores,
        'RMSE_Scores': rmse_scores
    }
    
    return cv_results


def print_cv_results(cv_results, model_name="Model"):
    """Print cross-validation results."""
    print(f"\n{model_name} - Cross-Validation Results (5-Fold):")
    print(f"  R² Score:  {cv_results['R2_Mean']:.4f} ± {cv_results['R2_Std']:.4f}")
    print(f"  RMSE:      {cv_results['RMSE_Mean']:.2f} ± {cv_results['RMSE_Std']:.2f}")