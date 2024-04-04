# Decision Tree Classifier for Social Network Ads

This Python script implements a Decision Tree Classifier to predict whether a user will purchase a product based on their age and estimated salary. The dataset used in this script is "Social_Network_Ads.csv".

## Prerequisites
- Python 3
- Required Python libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`

## Usage
1. Clone this repository or download the `Social_Network_Ads.csv` and `decision_tree_classifier.py` files.
2. Ensure that you have the necessary Python libraries installed (see Prerequisites).
3. Run the `decision_tree_classifier.py` script using a Python interpreter.
4. The script will train the Decision Tree Classifier on the training data, make predictions on the test data, and visualize the decision boundaries for both the training and test sets.

## File Descriptions
- `decision_tree_classifier.py`: The Python script containing the implementation of the Decision Tree Classifier.
- `Social_Network_Ads.csv`: The dataset used for training and testing the classifier.

## Dataset Description
The dataset contains the following columns:
- `User ID`: Unique identifier for each user.
- `Gender`: Gender of the user (Male or Female).
- `Age`: Age of the user.
- `Estimated Salary`: Estimated salary of the user.
- `Purchased`: Whether the user purchased the product (0 for not purchased, 1 for purchased).
