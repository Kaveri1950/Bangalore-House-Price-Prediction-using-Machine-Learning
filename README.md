# Bangalore House Price Prediction using Machine Learning

## Overview

This project is a **data science regression pipeline** that predicts **housing prices in Bangalore** using modern machine learning models, including stacking ensembles and neural networks.  
The workflow covers:  
- Robust data cleaning  
- Feature engineering  
- Dimensionality reduction for categories  
- Outlier analysis  
- Extensive EDA visualizations  
- Multiple regression models with grid search  
- Ensemble methods (Stacking, XGBoost) & Neural Network implementation  
- Saved model for API deployment

## Dataset

Source: [Bengaluru House Prices Dataset on Kaggle](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data)

Key features:
- area_type: Type of area (Super built-up, Plot, etc.)
- availability: Ready To Move/date
- location: Bangalore locality  
- size: BHK/bedrooms  
- society: Society/complex name  
- total_sqft: Area in sq.ft  
- bath: Bathrooms  
- balcony: Balconies  
- price: Price (lakhs)

## Data Preparation & Cleaning

- Unnecessary columns dropped, missing values handled.
- Bedroom/BHK feature extracted from raw text.
- Square foot cleaned by handling ranges (e.g., 2100–2850) and odd formats.  
- New feature `price_per_sqft` calculated.
- Dimensionality reduction on locations: rare locations tagged "other".
- Outliers removed based on inconsistent room/area/bathroom data, standard deviation, and price-per-sqft anomalies.

## Feature Engineering

- Minimum sq.ft per BHK set to 300 for realistic properties.
- One-hot encoding for location (reduces feature space).
- Final feature table: Numeric features + location dummies.

## Exploratory Data Analysis (EDA)

- Scatter plots: Price vs Sq.Ft, BHK vs Price for specific localities.
- Histograms: Price per sq.ft, bathroom counts.
- Correlation heatmap of all numerical features.
- Outlier visualization and removal steps.

## Model Building & Evaluation

### Individual Models

Trained and compared:
- Linear Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regression (SVR, with scaling)
- XGBoost Regressor

| Model               | Test R² Score | CV Mean R² |
|---------------------|--------------:|-----------:|
| Linear Regression   |   ~0.80       |   ~0.85    |
| Lasso Regression    |   ~0.69       |   ~0.73    |
| Decision Tree       |   ~0.65       |   ~0.72    |
| Random Forest       |   ~0.70       |   ~0.78    |
| SVM                 |   ~0.79       |   ~0.84    |
| XGBoost             |   ~0.67       |   --       |

### Stacking Ensemble Models

Stacked and blended the above base models using scikit-learn and XGBoost:

- StackingRegressor with base models: LinearRegression, Lasso, DecisionTree, RF, SVM, KNN, XGB.
- Meta-model: LinearRegression and XGBoost (for nonlinear stacking).
- Achieved R² scores up to **0.79** (ensemble), reducing RMSE and MAE.

### Deep Learning (Neural Networks)

- Built Keras/Tensorflow regression networks.
- Applied advanced callbacks (early stopping, ReduceLROnPlateau).
- Achieved **R² up to 0.86**, MAE ~15, RMSE ~36 (best configuration).


## Model Export & Prediction API

- Exported best model with pickle:  
  `banglore_home_prices_model.pickle`
- Saved columns/features with  
  `columns.json`
- Provided a universal `predict_price()` API to access predictions for any location/BHK/bath/sq.ft.

## Usage

1. Install required libraries:

    ```bash
    pip install numpy pandas scikit-learn xgboost matplotlib tensorflow
    ```

2. Clone/download repo and Kaggle dataset.

3. Train/evaluate:
    - Run Jupyter notebook or Python scripts.

4. Predict with API:

    ```python
    price = predict_price("JP Nagar", sq_ft, bath, bhk, model)
    ```

5. Deploy: Use exported pickle and columns.json for backend or web API.

## Example Visualizations

- Price vs Sq.Ft scatter: `plt.scatter(df['total_sqft'], df['price'])`
- Location-wise price-per-sqft barplots
- Histogram of price_per_sqft, bathroom count
- Heatmaps of feature correlation.
