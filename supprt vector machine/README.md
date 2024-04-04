# Support Vector Machine (SVM) for Classifying Apples vs Oranges

This Python script implements a Support Vector Machine (SVM) classifier to classify fruits as either apples or oranges based on their weights (in grams) and sizes (in centimeters). The dataset used in this script is named `data.csv`.

## Prerequisites
- Python 3
- Required Python libraries: `pandas`, `scikit-learn`, `numpy`, `matplotlib`

## Usage
1. Clone this repository or download the `data.csv` and `svm_classifier.py` files.
2. Ensure that you have the necessary Python libraries installed (see Prerequisites).
3. Run the `svm_classifier.py` script using a Python interpreter.
4. The script will split the dataset into training and test sets, train the SVM classifier on the training data, visualize the decision boundaries for both the training and test sets, and display the predictions.

## File Descriptions
- `svm_classifier.py`: The Python script containing the implementation of the Support Vector Machine classifier.
- `data.csv`: The dataset containing information about fruits' weights, sizes, and types.

## Dataset Description
The dataset contains the following columns:
- `Weight`: Weight of the fruit in grams.
- `Size`: Size of the fruit in centimeters.
- `Type`: Type of the fruit (`Apple` or `Orange`).

## Visualization
The script visualizes the decision boundaries of the SVM classifier on both the training and test sets, as well as the predictions made by the classifier.
