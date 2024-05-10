# Name: Kristy Nguyen & Thomas Bryant
# NSHE: 5006243601 | 2000193948
# Class: CS422-1001
# Assignment: Assignment 3
# Description: This project will import a dataset that has more than
# 3 continuous, real-valued, features and categorize them in a binary
# output. We used the 'Breast Cancer Wisconsin (Diagnostic) Data Set'
# for this project. It will then make a confusion matrix and print
# a classification report of the metrics on how the model performed.
# Source: https://data.world/health/breast-cancer-wisconsin

# Importing appropriate libraries
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score, log_loss

# Getting the directory and opening data file
# NOTE: both program and data file MUST be in the same directory
current_directory = os.path.dirname(os.path.realpath(__file__))
file_name = 'data.csv'
file_path = os.path.join(current_directory, file_name)
data = pd.read_csv(file_path)

# Drop last column incorrectly producing NaN values
data = data.dropna(axis=1, how='all')

# Test read with printing the data
print(data)

# Assuming 'diagnosis' is the name of the target column, set features (X) and outputs (Y)
y = data['diagnosis']
X = data.drop('diagnosis', axis=1)

# Split data into training (80%) and testing (20%) sets with a random state (31)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=83)

# Preprocess the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the ML Model
model = GaussianNB()
model.fit(X_train, y_train)
train_predictions = model.predict(X_train)

# Calculate and print metrics on the training set
for label in ['B', 'M']:
    indices = (y_train == label)
    y_train_binary = (y_train == label)
    train_predictions_binary = (train_predictions == label)
    tn, fp, fn, tp = confusion_matrix(y_train_binary, train_predictions_binary).ravel()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    print(f"\nMetrics for class '{label}' on the Training Set:")
    print(f"Accuracy for '{label}': {accuracy_score(y_train[indices], train_predictions[indices])}")
    print(f"Sensitivity for '{label}': {recall_score(y_train[indices], train_predictions[indices], pos_label=label, zero_division='warn')}")
    print(f"Specificity for '{label}': {specificity}")
    print(f"F1 Score for '{label}': {f1_score(y_train[indices], train_predictions[indices], pos_label=label)}")

# Test the model
y_pred = model.predict(X_test)
test_predictions = model.predict(X_test)

# Calculate and print metrics on the test set
for label in ['B', 'M']:
    indices = (y_test == label)
    y_test_binary = (y_test == label)
    test_predictions_binary = (test_predictions == label)
    tn, fp, fn, tp = confusion_matrix(y_test_binary, test_predictions_binary).ravel()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    print(f"\nMetrics for class '{label}' on the Test Set:")
    print(f"Accuracy for '{label}': {accuracy_score(y_test[indices], test_predictions[indices])}")
    print(f"Sensitivity for '{label}': {recall_score(y_test[indices], test_predictions[indices], pos_label=label, zero_division='warn')}")
    print(f"Specificity for '{label}': {specificity}")
    print(f"F1 Score for '{label}': {f1_score(y_test[indices], test_predictions[indices], pos_label=label)}")
    print(f"Log Loss for '{label}': {log_loss(y_test_binary, model.predict_proba(X_test)[:, model.classes_ == label])}")
