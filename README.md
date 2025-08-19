# Bangalore House Price Prediction using Machine Learning
#### This code is inspired by youtube tutorial codebasics

## Overview
This project is a **data science regression analysis** that aims to predict **housing prices in Bangalore** using machine learning techniques.  
Real estate pricing depends on multiple factors such as location, property size, number of bathrooms, and amenities. This project explores how these features influence property prices and builds a regression model that can provide reasonably accurate predictions.  

The workflow includes:
- **Data Collection**: The dataset is taken from Kaggle and contains details of various properties in Bangalore.  
- **Data Cleaning**: Handling missing values, standardizing inconsistent formats, and removing irrelevant columns.  
- **Feature Engineering**:  
  - Converting textual features into numerical values  
  - Reducing dimensionality by grouping rare locations  
  - Handling outliers in `total_sqft` and `price`  
- **Exploratory Data Analysis (EDA)**: Visualizing the distribution of prices, area types, availability trends, and relationship between features.  
- **Model Building**: Training regression models (e.g., Linear Regression, Lasso, Decision Tree) to predict property prices.  
- **Model Evaluation**: Measuring performance using metrics such as **R² score** and comparing results across models.  
- **Prediction**: The final model allows users to input property details (location, BHK, sqft, bathrooms) and get a predicted price.  

This project demonstrates the **end-to-end machine learning pipeline**: from raw dataset → preprocessing → feature engineering → model building → prediction.


## Dataset
The dataset used in this project is taken from **Kaggle**:  
[Bengaluru House Prices Dataset](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data)

It contains the following features:

- **area_type** → Type of the area (e.g., Super built-up, Plot, Built-up, Carpet)  
- **availability** → When the property is available (Ready To Move / Date)  
- **location** → Location in Bangalore  
- **size** → Number of BHK/Bedrooms  
- **society** → Name of the society/complex (if available)  
- **total_sqft** → Total area (in square feet)  
- **bath** → Number of bathrooms  
- **balcony** → Number of balconies  
- **price** → Price of the property (in lakhs)  



## Example Visualizations

The notebook includes several plots to better understand the dataset and the relationships between features:

- **Price vs Square Feet**: Scatter plot showing how house prices scale with area.  
- **Price Distribution by Location**: Average price per square foot for top locations in Bangalore.  
- **BHK vs Price**: Distribution of prices for different bedroom counts.  
- **Correlation Heatmap**: Showing relationships between numerical features (`price`, `bath`, `total_sqft`, etc.).  


