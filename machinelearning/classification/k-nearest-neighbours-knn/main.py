print('Step 1 - Importing the libraries')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('\nStep 2 - Importing the dataset')
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print('\nStep 3 - Splitting the dataset into the Training set and Test set')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print('\nStep 4 - Feature Scaling')
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print('\nStep 5 - Training the K-NN model on the Training set')
from sklearn.neighbors import KNeighborsClassifier
# Create an instance of the KNeighborsClassifier with the following parameters:
# - n_neighbors=5: Specifies the number of nearest neighbors (k) to consider for classification.
# - p=2: Specifies the distance metric to be used. p=2 corresponds to the Euclidean distance.
# (If p=1, it would use Manhattan distance instead.)
classifier = KNeighborsClassifier(n_neighbors = 5, p=2)
classifier.fit(X_train, y_train)

print('\nStep 6 - Predicting a new result')
print(classifier.predict(sc.transform([[30, 87000]])))

print('\nStep 7 - Predicting the Test set results')
y_predict = classifier.predict(X_test)
print(np.concatenate((y_test.reshape(len(y_test), 1), y_predict.reshape(len(y_test), 1)), 1))

print('\nStep 8 - Making the Confusion Matrix')
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_predict)
print(cm)
accuracy_score = accuracy_score(y_test, y_predict)
print(accuracy_score)

step = 0.25

print('\nStep 9 - Visualizing the Training Set Results')
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_train), y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = step),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = step))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

print('\nStep 10 - Visualizing the Test Set Results')
from matplotlib.colors import ListedColormap
X_set, y_set = sc.inverse_transform(X_test), y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 10, stop = X_set[:, 0].max() + 10, step = step),
                     np.arange(start = X_set[:, 1].min() - 1000, stop = X_set[:, 1].max() + 1000, step = step))
plt.contourf(X1, X2, classifier.predict(sc.transform(np.array([X1.ravel(), X2.ravel()]).T)).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('K-NN (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()