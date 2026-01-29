"""
visualization.py - Plotting and visualization functions

This module provides:
1. Feature importance plots
2. Correlation heatmaps
3. Price distribution plots
4. Model comparison visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_importance(model, feature_names, top_n=15, model_name="Model"):
    """
    Plot feature importance from trained model.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_names : list
        Names of features
    top_n : int
        Number of top features to display
    model_name : str
        Name of the model for title
    """
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        raise ValueError("Model must have feature_importances_ or coef_ attribute")
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
    plt.xlabel('Importance Score')
    plt.title(f'{model_name} - Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return importance_df


def plot_feature_importance_percentage(model, feature_names, top_n=15, model_name="Model"):
    """
    Plot feature importance as percentage contribution.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model with feature_importances_ attribute
    feature_names : list
        Names of features
    top_n : int
        Number of top features to display
    model_name : str
        Name of the model for title
    """
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_)
    else:
        raise ValueError("Model must have feature_importances_ or coef_ attribute")
    
    # Convert to percentage
    importance_pct = (importance / importance.sum()) * 100
    
    # Create dataframe
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance_Pct': importance_pct
    }).sort_values('Importance_Pct', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
    bars = plt.barh(importance_df['Feature'], importance_df['Importance_Pct'], color=colors)
    plt.xlabel('Feature Importance (%)')
    plt.title(f'{model_name} - Top {top_n} Feature Contribution')
    
    # Add percentage labels
    for i, (idx, row) in enumerate(importance_df.iterrows()):
        plt.text(row['Importance_Pct'] + 0.2, i, f"{row['Importance_Pct']:.1f}%", va='center')
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    return importance_df


def plot_correlation_matrix(df, figsize=(12, 10)):
    """
    Plot correlation heatmap for numerical features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with numerical features
    figsize : tuple
        Figure size
    """
    # Select numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation
    corr = numerical_df.corr()
    
    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix - Numerical Features')
    plt.tight_layout()
    plt.show()


def plot_price_distribution(prices, title="Price Distribution", currency_symbol="$"):
    """
    Plot distribution of prices.
    
    Parameters:
    -----------
    prices : pd.Series or array-like
        Price values
    title : str
        Plot title
    currency_symbol : str
        Currency symbol for labeling
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Remove NaN values
    prices_clean = prices.dropna() if hasattr(prices, 'dropna') else prices[~np.isnan(prices)]
    
    # Histogram
    axes[0, 0].hist(prices_clean, bins=50, edgecolor='black', color='steelblue')
    axes[0, 0].set_xlabel(f'Price ({currency_symbol})')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Box plot
    axes[0, 1].boxplot(prices_clean)
    axes[0, 1].set_ylabel(f'Price ({currency_symbol})')
    axes[0, 1].set_title('Box Plot')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Log-scale histogram
    axes[1, 0].hist(np.log1p(prices_clean), bins=50, edgecolor='black', color='orange')
    axes[1, 0].set_xlabel(f'Log(Price)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Log-Transformed Distribution')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Summary statistics
    axes[1, 1].axis('off')
    stats_text = f"""
    Statistics:
    
    Mean:       {np.mean(prices_clean):,.0f}
    Median:     {np.median(prices_clean):,.0f}
    Std Dev:    {np.std(prices_clean):,.0f}
    Min:        {np.min(prices_clean):,.0f}
    Max:        {np.max(prices_clean):,.0f}
    Q1:         {np.percentile(prices_clean, 25):,.0f}
    Q3:         {np.percentile(prices_clean, 75):,.0f}
    IQR:        {np.percentile(prices_clean, 75) - np.percentile(prices_clean, 25):,.0f}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5))
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_model_comparison(comparison_df):
    """
    Plot comparison of multiple models.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        Dataframe with model metrics from model_evaluation.compare_models()
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # R² Score comparison
    axes[0].barh(comparison_df['Model'], comparison_df['R2_Score'], color='steelblue')
    axes[0].set_xlabel('R² Score')
    axes[0].set_title('Model Comparison - R² Score')
    axes[0].set_xlim([0, 1])
    for i, v in enumerate(comparison_df['R2_Score']):
        axes[0].text(v + 0.02, i, f'{v:.4f}', va='center')
    
    # RMSE comparison
    axes[1].barh(comparison_df['Model'], comparison_df['RMSE'], color='coral')
    axes[1].set_xlabel('RMSE')
    axes[1].set_title('Model Comparison - RMSE (lower is better)')
    for i, v in enumerate(comparison_df['RMSE']):
        axes[1].text(v + 1000, i, f'{v:,.0f}', va='center')
    
    plt.tight_layout()
    plt.show()


def plot_market_comparison(ames_metrics, malaysia_metrics):
    """
    Plot comparison between Ames and Malaysia models.
    
    Parameters:
    -----------
    ames_metrics : dict
        Metrics from Ames model
    malaysia_metrics : dict
        Metrics from Malaysia model
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # R² Score comparison
    models = ['Ames\n(USA)', 'Malaysia']
    r2_scores = [ames_metrics['R2_Score'], malaysia_metrics['R2_Score']]
    colors = ['steelblue', 'orange']
    
    axes[0].bar(models, r2_scores, color=colors, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('Cross-Market Model Performance: R² Score')
    axes[0].set_ylim([0, 1])
    for i, v in enumerate(r2_scores):
        axes[0].text(i, v + 0.05, f'{v:.3f}', ha='center', fontweight='bold', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Accuracy gap
    accuracy_gap = r2_scores[0] - r2_scores[1]
    feature_gap = [80, 7]
    
    axes[1].scatter(feature_gap, r2_scores, s=300, c=colors, edgecolor='black', linewidth=2, alpha=0.7)
    axes[1].plot(feature_gap, r2_scores, 'k--', alpha=0.3)
    axes[1].set_xlabel('Number of Features')
    axes[1].set_ylabel('R² Score')
    axes[1].set_title('Data Granularity vs Model Accuracy')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3)
    
    # Add labels
    axes[1].text(80, r2_scores[0] + 0.03, 'Ames\n(High Granularity)', ha='center', fontweight='bold')
    axes[1].text(7, r2_scores[1] - 0.08, 'Malaysia\n(Limited Data)', ha='center', fontweight='bold')
    
    # Add annotation for gap
    mid_x = (80 + 7) / 2
    mid_y = (r2_scores[0] + r2_scores[1]) / 2
    axes[1].annotate('', xy=(80, r2_scores[0]), xytext=(7, r2_scores[1]),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=2, alpha=0.5))
    axes[1].text(mid_x, mid_y + 0.05, f'Gap: {accuracy_gap:.3f}', 
                ha='center', color='red', fontweight='bold', fontsize=10)
    
    plt.suptitle('Ames vs Malaysia: The Impact of Data Quality', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()