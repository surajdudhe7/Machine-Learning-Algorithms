# Logistic Regression for Social Network Ads

This Python script implements Logistic Regression to predict whether a user will purchase a product based on their age and estimated salary. The dataset used in this script is "Social_Network_Ads.csv".

## Prerequisites
- Python 3
- Required Python libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`

## Usage
1. Clone this repository or download the `Social_Network_Ads.csv` and `logistic_regression.py` files.
2. Ensure that you have the necessary Python libraries installed (see Prerequisites).
3. Run the `logistic_regression.py` script using a Python interpreter.
4. The script will train the Logistic Regression model on the training data, make predictions on the test data, visualize the decision boundaries for the training set, and display the results.

## File Descriptions
- `logistic_regression.py`: The Python script containing the implementation of Logistic Regression.
- `Social_Network_Ads.csv`: The dataset containing information about users' age, estimated salary, and purchase history.

## Dataset Description
The dataset contains the following columns:
- `Age`: Age of the user.
- `Estimated Salary`: Estimated salary of the user.
- `Purchased`: Whether the user purchased the product (0 for not purchased, 1 for purchased).

## Visualization
The script visualizes the decision boundaries of the Logistic Regression model on the training set.
