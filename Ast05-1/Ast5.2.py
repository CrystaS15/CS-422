# Name: Kristy Nguyen & Thomas Bryant
# NSHE: 5006243601 | 2000193948
# Class: CS422-1001
# Assignment: Assignment 5
# Description: This project will import a dataset that has more than
# 3 continuous, real-valued, features and categorize them in a
# continuous real-valued output. We used the 'Diabetes Data Set' from
# scikit learn for this project. It will then use the SGDRegressor and
# LinearRegression to make an ordinary least squares (OLS) ML model
# to train and test two separate ML models and compare their metrics.
# Finally, it will print a report of the metrics on how well both
# models performed on both the training and test data sets.
# Data Set Source: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=83)

# Add a column of ones to the feature matrix for the intercept term
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Set, Train, & Test Ordinary Least Squares (OLS) from scratch
XTX_inv = np.linalg.inv(np.matmul(X_train.T, X_train))
w_ols = np.matmul(np.matmul(XTX_inv, X_train.T), y_train)
y_pred_train_ols = np.matmul(X_train, w_ols)
y_pred_test_ols = np.matmul(X_test, w_ols)

# Stochastic Gradient Descent (SGD)
sgd_model = SGDRegressor(max_iter=10000, random_state=83)
sgd_model.fit(X_train, y_train)
y_pred_train_sgd = sgd_model.predict(X_train)
y_pred_test_sgd = sgd_model.predict(X_test)

# Evaluate OLS model on training set
mse_train_ols = mean_squared_error(y_train, y_pred_train_ols)
mae_train_ols = mean_absolute_error(y_train, y_pred_train_ols)
r2_train_ols = r2_score(y_train, y_pred_train_ols)

print("Results for Ordinary Least Squares (OLS) on Training Set:")
print(f'Mean Squared Error: {mse_train_ols}')
print(f'Mean Absolute Error: {mae_train_ols}')
print(f'R-squared: {r2_train_ols}')
print(f'\nSolution \'w\' Parameter Vector (OLS):\n {w_ols}')

# Evaluate OLS model on test set
mse_test_ols = mean_squared_error(y_test, y_pred_test_ols)
mae_test_ols = mean_absolute_error(y_test, y_pred_test_ols)
r2_test_ols = r2_score(y_test, y_pred_test_ols)

print("\nResults for Ordinary Least Squares (OLS) on Test Set:")
print(f'Mean Squared Error: {mse_test_ols}')
print(f'Mean Absolute Error: {mae_test_ols}')
print(f'R-squared: {r2_test_ols}')

# Evaluate SGD model on training set
mse_train_sgd = mean_squared_error(y_train, y_pred_train_sgd)
mae_train_sgd = mean_absolute_error(y_train, y_pred_train_sgd)
r2_train_sgd = r2_score(y_train, y_pred_train_sgd)

print("\nResults for Stochastic Gradient Descent (SGD) on Training Set:")
print(f'Mean Squared Error: {mse_train_sgd}')
print(f'Mean Absolute Error: {mae_train_sgd}')
print(f'R-squared: {r2_train_sgd}')
print(f'\nSolution \'w\' Parameter Vector (SGD):\n {sgd_model.coef_}')

# Evaluate SGD model on test set
mse_test_sgd = mean_squared_error(y_test, y_pred_test_sgd)
mae_test_sgd = mean_absolute_error(y_test, y_pred_test_sgd)
r2_test_sgd = r2_score(y_test, y_pred_test_sgd)

print("\nResults for Stochastic Gradient Descent (SGD) on Test Set:")
print(f'Mean Squared Error: {mse_test_sgd}')
print(f'Mean Absolute Error: {mae_test_sgd}')
print(f'R-squared: {r2_test_sgd}')
