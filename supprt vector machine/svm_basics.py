# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

# Read data from CSV file
data = pd.read_csv('data.csv')

# Split data into training and test set
training_set, test_set = train_test_split(data, test_size=0.2, random_state=1)
#print(x_train,y_train)
#print(x_test,y_test)


# Prepare data for applying it to SVM
x_train = training_set.iloc[:, 0:2].values  # Features
y_train = training_set.iloc[:, 2].values    # Target
x_test = test_set.iloc[:, 0:2].values       # Features
y_test = test_set.iloc[:, 2].values         # Target

# Fit the data (train a model)
classifier = SVC(kernel='rbf', random_state=1, C=1, gamma='auto')
classifier.fit(x_train, y_train)

# Perform prediction on x_test data
y_pred = classifier.predict(x_test)
#test_set['prediction']=y_pred
#print(y_pred)


# Creating confusion matrix and calculating accuracy
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
accuracy = float(cm.diagonal().sum()) / len(y_test)
print('Model accuracy is:', accuracy * 100, '%')
