# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
datasets = pd.read_csv('Social_Network_Ads.csv')
X = datasets.iloc[:, [2, 3]].values
Y = datasets.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

# Fitting the classifier into the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_Train, Y_Train)

# Predicting the Test set results
Y_Pred = classifier.predict(X_Test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_Test, Y_Pred)

# Visualizing the Training set results
from matplotlib.colors import ListedColormap
plt.figure(figsize=(10, 6))

X1, X2 = np.meshgrid(np.linspace(X_Train[:, 0].min() - 1, X_Train[:, 0].max() + 1, 100),
                     np.linspace(X_Train[:, 1].min() - 1, X_Train[:, 1].max() + 1, 100))
boundary = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
plt.contourf(X1, X2, boundary, alpha=0.75, cmap=ListedColormap(('red', 'green')))

plt.scatter(X_Train[Y_Train == 0, 0], X_Train[Y_Train == 0, 1], color='red', label='0 (Not Purchased)')
plt.scatter(X_Train[Y_Train == 1, 0], X_Train[Y_Train == 1, 1], color='green', label='1 (Purchased)')

plt.title('Decision Tree Classifier (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

# Visualizing the Test set results
plt.figure(figsize=(10, 6))

X1, X2 = np.meshgrid(np.linspace(X_Test[:, 0].min() - 1, X_Test[:, 0].max() + 1, 100),
                     np.linspace(X_Test[:, 1].min() - 1, X_Test[:, 1].max() + 1, 100))
boundary = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
plt.contourf(X1, X2, boundary, alpha=0.75, cmap=ListedColormap(('red', 'green')))

plt.scatter(X_Test[Y_Test == 0, 0], X_Test[Y_Test == 0, 1], color='red', label='0 (Not Purchased)')
plt.scatter(X_Test[Y_Test == 1, 0], X_Test[Y_Test == 1, 1], color='green', label='1 (Purchased)')

plt.title('Decision Tree Classifier (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
