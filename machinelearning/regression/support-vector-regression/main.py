print('Step 1 - Importing the libraries')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('\nStep 2 - Importing the dataset')
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print(X)
print(y)

print('\nStep 3 - Transforming y dataset into a 2D array as StandardScalar expects 2D array')
y = y.reshape(len(y), 1)

print('\nStep 4 - Feature Scaling')
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# If we use the same scaler (sc_X) for y (target):
# 1. During `sc_X.fit_transform(X)`:
#    - sc_X computes the mean and standard deviation of X and stores them internally.
# 2. During `sc_X.fit_transform(y)`:
#    - sc_X re-computes the mean and standard deviation, but this time for y.
#    - This overwrites the mean and standard deviation of X with those of y.
#    - As a result, sc_X now reflects the statistics of y, not X.
# 3. Later, if we use sc_X to perform inverse transformations or further scaling on X:
#    - The transformations would be incorrect because sc_X no longer contains the correct
#      mean and variance for X.

sc_y = StandardScaler()
y = sc_y.fit_transform(y)
print(X)
print(y)

print('\nStep 5 - Training the SVR on the whole dataset')
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

print('\nStep 6 - Predicting a new result')
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1, 1))
print(y_pred)

print('\nStep 7 - Visualising the SVR results')
inverse_transform_X = sc_X.inverse_transform(X)
inverse_transform_y = sc_y.inverse_transform(y)
plt.scatter(inverse_transform_X, inverse_transform_y, color='red')
plt.plot(inverse_transform_X, sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

print('\nStep 8 - Visualising the SVR results (for higher resolution and smoother curve)')
# Create a 1D grid of values for plotting or further analysis
# Step 1: Flatten the inverse-transformed feature matrix X to a 1D array using ravel
# ravel() flattens the 2D array into a 1D array, as np.arange expects a 1D sequence
# Step 2: Find the minimum value in the flattened array
# This will give the start of the range for generating the grid
# Step 3: Find the maximum value in the flattened array
# This will give the end of the range for generating the grid
# Step 4: Use np.arange to create a grid of values between the minimum and maximum values
# The range is from min_value to max_value with a step size of 0.01
# Now, X_grid contains a 1D array with values between min_value and max_value
# The grid is useful for visualizing or making predictions across the feature range
X_grid = np.arange(min(inverse_transform_X.ravel()), max(inverse_transform_X.ravel()), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))  # Reshape to 2D for compatibility
plt.scatter(inverse_transform_X, inverse_transform_y, color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()