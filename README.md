# 🏠 House Price Prediction using Ridge & Lasso Regression

This project involves building a predictive model to estimate house prices using real-world data. The dataset is taken from the [Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques), and we implement regularized regression techniques (Ridge & Lasso) to minimize prediction error and prevent overfitting.

---

## 📊 Dataset
- **Source:** [Kaggle Housing Dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)
- **Size:** ~1,400 training records with 80 features
- **Target:** `SalePrice` of residential houses in Ames, Iowa

---

## 🧠 Objective
Predict house sale prices by:
- Cleaning and preprocessing structured data
- Engineering meaningful numerical & categorical features
- Applying Ridge and Lasso regression with cross-validation
- Measuring prediction accuracy via RMSE

---

## 🛠️ Techniques Used

- **Data Cleaning & Preprocessing:**  
  - Dropped high-missing-value columns
  - One-hot encoded categorical variables
  - Standardized numerical variables

- **Feature Engineering:**  
  - Transformed skewed variables using log transformations  
  - Removed outliers with extreme values in numeric columns

- **Modeling:**  
  - Applied **Ridge** (L2) and **Lasso** (L1) regularization
  - Tuned alpha parameter manually and via cross-validation (5-fold)
  - Evaluated models using RMSE on hold-out test set

---


