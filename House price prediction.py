# -*- coding: utf-8 -*-
"""

@author: barath
"""

import numpy as nm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import skew
from sklearn.linear_model import RidgeCV,LassoCV
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('train.csv')                             # Import the dataset
print(data.shape)
print(data.head())

# Drop columns with more than 50% null values
null_threshold =0.5
missing_frac=data.isnull().mean()
data=data.drop(columns=missing_frac[missing_frac > null_threshold].index)   # Drop columns with too many missing values 


# Removing outliers using quantile capping (1st to 99th percentile)
num_cols=data.select_dtypes(include=[nm.number]).columns                    # Identify Numeric columns

for col in num_cols:
    low=data[col].quantile(0.01)
    up=data[col].quantile(0.99)
    data=data[(data[col]>=low) & (data[col]<=up) | (data[col].isna())]

data = data.fillna(data.median(numeric_only=True))

# Log-transform features with skew > 0.75
skewed_feats=data[num_cols].apply(lambda x: skew(x.dropna()))
high_skew = skewed_feats[skewed_feats > 0.75].index
data[high_skew] = nm.log1p(data[high_skew])


#Seperate the target and features
y=data['SalePrice']
X=data.drop(columns=['SalePrice','Id'])                    
X=pd.get_dummies(X)

# 5 Fold CV (80-20 split)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)   #Maintanis consistency in results when running experiments multiple times

# Standardize Numeric Features
scalar=StandardScaler()
X_train=scalar.fit_transform(X_train)
X_test=scalar.transform(X_test)

# Train Models
alphas=nm.logspace(-3,3,100)

ridge_cv=RidgeCV(alphas=alphas,scoring='neg_mean_squared_error',cv=5)
ridge_cv.fit(X_train,y_train)

lasso_cv=LassoCV(alphas=alphas,cv=5,max_iter=10000)
lasso_cv.fit(X_train,y_train)

# Prediction

ridge_pred=ridge_cv.predict(X_test)
lasso_pred=lasso_cv.predict(X_test)

ridge_rmse=nm.sqrt(mean_squared_error(y_test,ridge_pred))
lasso_rmse=nm.sqrt(mean_squared_error(y_test,ridge_pred))

print('The Root Mean Squared Error are:')
print('Ridge RMSE =',ridge_rmse)
print('Lasso RMSE =',lasso_rmse)

