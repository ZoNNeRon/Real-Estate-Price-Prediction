# Real Estate Price Prediction: A Comparative ML Study

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) 
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) 
![Model Accuracy](https://img.shields.io/badge/Accuracy-86.4%25_US_%7C_57.5%25_MY-brightgreen)
![Status](https://img.shields.io/badge/Status-Complete-success)

## Executive Summary

This project evaluates **three machine learning algorithms** (Lasso, Random Forest, XGBoost) for housing price prediction across **two markets with contrasting data structures**: 
- **Ames, Iowa (USA)**: Mature market with granular, individual-level data (82 attributes)
- **Malaysia (2025)**: Emerging market with aggregated, township-level data (7 key features)

**Key Finding:** Data quality and granularity drive model performance more than algorithm complexity. Achieved **R² = 0.864** (±12.47% error) on U.S. market vs. **R² = 0.575** (±34.72% error) on Malaysian market—demonstrating market-specific models outperform universal approaches.

---

## Project Goals

1. **Compare ML algorithms** across markets with different data characteristics
2. **Identify price drivers** for each market (structural attributes vs. location)
3. **Quantify the impact** of data granularity on prediction accuracy
4. **Build an interactive application** for real-time price estimation

---

## Repository Structure

```
Real-Estate-Price-Prediction/
├── README.md                              # Project description
├── requirements.txt                       # Python dependencies
├── LICENSE                               # MIT License
│
├── data/
│   ├── ames/
│   │   ├── AmesHousing.csv             # 2,930 house sales with 80 features
│   │   └── data_dictionary.md           # Feature descriptions
│   ├── malaysia/
│   │   ├── malaysia_house_price_data_2025.csv  # 1,946 townships
│   │   └── data_dictionary.md
│   └── processed/                        # Preprocessed & feature-engineered data
│
├── notebooks/                            # Jupyter notebooks with analysis
│   ├── 00_Data_Loading_and_EDA.ipynb    # Initial exploration
│   ├── 01_Ames_Exploratory_Analysis.ipynb
│   ├── 02_Malaysia_Exploratory_Analysis.ipynb
│   ├── 03_Data_Preprocessing.ipynb       # Cleaning & imputation
│   ├── 04_Feature_Engineering.ipynb
│   ├── 05_Model_Training_Ames.ipynb      # Train Lasso, RF, XGBoost
│   ├── 06_Model_Training_Malaysia.ipynb
│   ├── 07_Feature_Importance_Analysis.ipynb
│   └── 08_Cross_Market_Comparison.ipynb  # Results & insights
│
├── src/                                  # Python modules
│   ├── __init__.py                   # Initialisation file
│   ├── data_loader.py                   # Data loading & preprocessing
│   ├── feature_engineering.py           # Feature creation
│   ├── model_training.py                # Training pipeline
│   ├── model_evaluation.py              # Metrics & analysis
│   └── visualization.py                 # Plotting functions
│
├── results/
│   ├── models/                          # Trained model files (.pkl)
│   │   ├── ames_lasso_model.pkl
│   │   ├── ames_rf_model.pkl
│   │   ├── ames_xgb_model.pkl
│   │   ├── malaysia_lasso_model.pkl
│   │   ├── malaysia_rf_model.pkl
│   │   └── malaysia_xgb_model.pkl
│   ├── visualizations/                  # Charts & plots
│   │   ├── ames_price_distribution.png
│   │   ├── malaysia_price_distribution.png
│   │   ├── ames_feature_importance.png
│   │   ├── malaysia_feature_importance.png
│   │   ├── model_performance_comparison.png
│   │   └── cross_market_analysis.png
│   └── model_metrics.csv                # Performance summary
│
├── app/                                 # Interactive web application
│   ├── index.html                       # Main interface (5 tabs)
│   ├── styles.css                       # Styling
│   └── script.js                        # Prediction logic
│
├── docs/                                # Documentation
      ├── RESEARCH_ARTICLE.pdf			# Project summary
```

---

## Quick Start

### Prerequisites
- Python 3.9+
- pip or conda

### Installation

1. **Clone repository**
```bash
git clone https://github.com/YOUR-USERNAME/Real-Estate-Price-Prediction.git
cd Real-Estate-Price-Prediction
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run Jupyter notebooks**
```bash
jupyter notebook
```

Start with `notebooks/00_Data_Loading_and_EDA.ipynb`

### Using the Interactive App

Open `app/index.html` in a web browser to:
- View dataset statistics and visualizations
- Compare model performance across markets
- Make real-time price predictions
- Explore market-specific price drivers

---

## Dataset Overview

| Aspect | Ames (USA) | Malaysia |
|--------|-----------|----------|
| **Observations** | 2,930 houses | 1,946 townships |
| **Features** | 82 attributes | 7 key variables |
| **Price Range** | $34.9K - $755K | RM 27K - RM 11.4M |
| **Avg Price** | $180,921 | RM 490,685 |
| **Time Period** | 2006-2010 | 2025 market |
| **Data Type** | Transaction-level | Aggregated township |

### Key Features

**Ames (USA) - Structural Focus:**
- OverallQual (house quality 1-10)
- GrLivArea (living area in sqft)
- GarageCars (garage capacity)
- TotalBsmtSF (basement area)
- YearBuilt (construction year)

**Malaysia - Location Focus:**
- MedianPSF (price per square foot)
- Area (township location)
- State (geographical region)
- Type (property category)
- Tenure (freehold/leasehold)

---

## Machine Learning Models

### Models Used

| Model | Type | Best For | Ames R² | Malaysia R² |
|-------|------|----------|---------|------------|
| **Lasso Regression** | Linear | Interpretability | 0.833 | 0.537 |
| **Random Forest** | Tree-Ensemble | Non-linear patterns | 0.847 | 0.510 |
| **XGBoost** | Gradient Boosting | Overall accuracy | **0.864** | **0.575** |

### Hyperparameter Tuning

**Lasso:**
- Alpha: Optimized via cross-validation (LassoCV)
- Uses StandardScaler for normalization

**Random Forest:**
- n_estimators: 100 trees
- max_depth: 15 (Ames), 10 (Malaysia)
- random_state: 42 (reproducibility)

**XGBoost:**
- max_depth: 6 (tree depth)
- learning_rate: 0.1
- n_estimators: 100 iterations
- Handles both linear and non-linear patterns

---

## Results & Findings

### Model Performance

#### Ames Housing Market (U.S.)
```
Model              RMSE      MAE        R²
─────────────────────────────────────────
Lasso              $35,738   $21,312    0.833
Random Forest      $34,308   $19,644    0.847
XGBoost            $32,280   $19,761    0.864 ✓ Best
```

**Interpretation:** XGBoost explains **86.4%** of price variance with **±12.47%** average error. Excellent for predicting individual home prices in this market.

#### Malaysia Housing Market
```
Model              RMSE        MAE        R²
──────────────────────────────────────────
Lasso              RM 285,485  RM 143,303 0.537
Random Forest      RM 293,729  RM 127,439 0.510
XGBoost            RM 273,477  RM 129,134 0.575 ✓ Best
```

**Interpretation:** XGBoost explains **57.5%** of price variance with **±34.72%** average error. Moderate accuracy due to aggregated township-level data with limited granular features.

### Key Insights

#### 1. **Data Granularity Drives Accuracy**
- U.S. dataset with 82 specific attributes → 86.4% R²
- Malaysia dataset with 7 aggregate features → 57.5% R²
- **Conclusion:** Feature richness, not algorithm sophistication, limits performance in emerging markets

#### 2. **Market-Specific Price Drivers**

**USA: Structural Attributes Dominate**
```
Top 3 Features (XGBoost):
1. OverallQual (16.9%)  - House quality
2. GarageCars (16.1%)   - Garage capacity
3. GrLivArea (12.3%)    - Living area
→ Americans pay premium for craftsmanship & practicality
```

**Malaysia: Location Dominance**
```
Top 3 Features (XGBoost):
1. MedianPSF (13.0%)    - Location premium
2. Area_KLCC (12.8%)    - KL proximity
3. State_Selangor (11.5%)
→ Malaysians pay for location access and development tier
```

#### 3. **Algorithm Selection Matters Conditionally**
- In data-rich environments: XGBoost gains ~3% over simpler models
- In data-sparse environments: Simpler Lasso and RF compete with XGBoost
- **Best Practice:** Match model complexity to data richness

---

## Methodology

### Data Preprocessing Pipeline

1. **Missing Value Handling**
   - Removed features with >50% missing data
   - Numeric: Median imputation
   - Categorical: Mode imputation

2. **Categorical Encoding**
   - One-hot encoding for categorical variables
   - Creates additional features (e.g., State_Selangor, Type_Condo)

3. **Feature Selection**
   - Univariate statistical selection (Ames: ~40 features selected from 80)
   - Removes low-variance or uncorrelated features

4. **Scaling & Normalization**
   - StandardScaler: (X - mean) / std
   - Essential for Lasso (L1 regularization sensitive to scale)
   - Applied separately to train/validation sets

### Model Training Pipeline

```python
# Pseudocode
1. Load data
   → X_train, X_test, y_train, y_test (80/20 split)

2. Preprocess
   → Handle missing values
   → Encode categorical variables
   → Scale features

3. Train models
   → Lasso (with cross-validation for alpha)
   → Random Forest (100 trees, max_depth optimized)
   → XGBoost (sequential boosting)

4. Evaluate
   → Metrics: R², RMSE, MAE
   → Feature importance extraction
   → Cross-market comparison

5. Visualize
   → Price distributions
   → Feature importance charts
   → Model performance comparison
   → Residual analysis
```

---

## Key Learnings

### For ML Engineers
1. **Data quality >> algorithm sophistication** - Clean, granular data beats complex models trained on sparse data
2. **Market context matters** - Same algorithm behaves differently across markets; tune independently
3. **Feature engineering crucial** - Location extraction in Malaysia showed 10-15% improvement potential
4. **Cross-validation essential** - Particularly for Lasso to prevent overfitting with correlated features

### For Real Estate Professionals
1. **U.S. market:** Use detailed property specs + XGBoost for high-accuracy valuations
2. **Malaysia market:** Supplement township-level data with external indicators (transit distance, school proximity)
3. **Market differences:** Can't use U.S. model directly on Malaysian data; location matters more there
4. **Data investment:** Collecting granular property-level data would improve Malaysian predictions by 20-30%

---

## Interactive Application Features

### 5 Main Tabs

#### 1. **Overview & Statistics**
- Dataset comparison (Ames vs Malaysia)
- Key metrics (R², RMSE, MAE)
- Model performance summary

#### 2. **Ames Market Deep Dive**
- Price distribution histogram
- Top 10 feature importance bar chart
- Market statistics (avg price, median, std dev)

#### 3. **Malaysia Market Deep Dive**
- Township-level price distribution
- Feature importance for Malaysia
- Location-based price breakdown

#### 4. **Cross-Market Comparison**
- Side-by-side R² comparison
- Feature importance differences
- Market characteristic matrix

#### 5. **Live Price Predictor**
- **USA Tab:**
  - Inputs: Quality (1-10), Living Area (sqft), Garage Capacity, Basement Quality
  - Output: Estimated price ± 12.47% confidence interval
  
- **Malaysia Tab:**
  - Inputs: Price/Sq Ft (RM), Property Type, Location Premium, Market Activity
  - Output: Estimated median township price ± 34.72% confidence interval

---

## How to Use

### For Model Exploration
1. Open `notebooks/00_Data_Loading_and_EDA.ipynb` to understand data
2. Follow notebooks in numerical order (01 → 08)
3. Each notebook is self-contained but builds on previous analysis

### For Price Prediction
1. Open `app/index.html` in web browser
2. Navigate to "Live Price Predictor" tab
3. Adjust sliders/inputs for your property
4. View estimated price with confidence range

### For Integration
```python
from src.model_training import ModelTrainer
import pickle

# Load trained model
with open('results/models/ames_xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Make prediction
prediction = model.predict([[quality, area, garage, ...]])
```

---

## Development & Testing

### Run Tests
```bash
pytest tests/
```

### Train Models from Scratch
```bash
# From project root
python -m src.model_training --dataset ames --algorithm xgb --save
```

### Generate Visualizations
```bash
python -m src.visualization --generate_all
```

---

## Academic References

This project was developed as part of the **CCS5600 Machine Learning course** at Universiti Putra Malaysia (UPM) and documented in:

- **IEEE Paper:** "Real Estate Price Prediction and Feature Analysis: A Comparative Study of Housing Models in U.S. and Malaysian Markets"
- **Presentation:** CCS5600-Presentation.pdf (included in `docs/`)

**Key References:**
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD 2016*
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32
- Rosen, S. (1974). Hedonic prices and implicit markets. *Journal of Political Economy*, 82(1), 34-55

---

## Limitations

1. **Malaysia Data Granularity**
   - Township-level aggregates limit individual property precision
   - Sparse features (7 vs 80 in USA) constrain model
   - Missing property-level details (condition, age, renovations)

2. **Temporal Scope**
   - Ames data: 2006-2010 (historical)
   - Malaysia data: 2025 snapshot
   - Models may not generalize to future market conditions

3. **Geographic Specificity**
   - Ames dataset: Iowa only (small state)
   - Malaysia: Nationwide aggregates
   - Results not directly transferable to other U.S. states or Asian markets

4. **Feature Engineering Constraints**
   - Malaysia type/tenure extraction simplified complex attributes
   - Median imputation assumes "average" for missing values
   - No interaction terms explicitly engineered

---

## Author

**Dmitrii Vasilov**
- Data Science Student | UPM
- LinkedIn: https://www.linkedin.com/in/dmitrii-vasilov-bb6a603a8/
- Email: DMVasilov@yandex.ru
- Location: Selangor, Malaysia

---

## License

MIT License - See LICENSE file for details

---

## Acknowledgments

- Dataset: Ames Housing (Data Science for Software Engineering course, Iowa State University)
- Advisors: Faculty of Computer Science, Universiti Putra Malaysia
- Team: Students of course CCS5600 “Machine Learning”
- Framework: scikit-learn, XGBoost, pandas, matplotlib