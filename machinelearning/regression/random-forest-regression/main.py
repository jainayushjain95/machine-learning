print('Step 1 - Importing the libraries')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('\nStep 2 - Importing the dataset')
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

print('\nStep 3 - Training the Random Forest Regression model on the whole dataset')
from sklearn.ensemble import RandomForestRegressor

# Create an instance of RandomForestRegressor with specific parameters:
# n_estimators=10: This means the model will build 10 decision trees in the random forest.
# random_state=0: This ensures reproducibility by fixing the random seed.
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, y)

print('\nStep 4 - Predicting a new result')
y_pred = regressor.predict([[6.5]])
print(y_pred)

print('\nStep 5 - Visualising the RFR results (for higher resolution)')
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()