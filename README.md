# linear_regression
Linear Regression: Linear Regression is a supervised learning algorithm that models the relationship between a dependent variable and one or more independent variables using a linear equation. It minimizes the difference between predicted and actual values using the least squares method. 
It is widely used for trend analysis, forecasting, and predictive modeling.
Features

Simple and interpretable model.

Used for trend analysis, forecasting, and predictive modeling.

Supports both simple and multiple regression.

Installation

Ensure you have Python installed, then install the required libraries:

pip install numpy pandas matplotlib scikit-learn

Usage

Clone the repository

git clone https://github.com/your-repo/linear_regression.git
cd linear_regression

Run the script

python linear_regression.py

Example Code:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Generate sample data
X = np.random.rand(100, 1) * 10
y = 2.5 * X + np.random.randn(100, 1) * 2

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Plot results
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predicted')
plt.legend()
plt.show()

