print('Step 1 - Importing the libraries')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('\nStep 2 - Importing the dataset')
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#Doing this to compare the results pf SLR and PLR models
print('\nStep 3 - Training the Linear Regression model on the whole dataset')
from sklearn.linear_model import LinearRegression
regressor_slr = LinearRegression()
regressor_slr.fit(X, y)


print('\nStep 4 - Training the Polynomial Regression model on the whole dataset')
from sklearn.preprocessing import PolynomialFeatures

# Initialize the PolynomialFeatures transformer with the desired degree.
# 'degree=2' means we are creating polynomial features up to the second degree (e.g., x, x^2).
regressor_poly = PolynomialFeatures(degree=2)

# Transform the input matrix of features, X, into a new matrix that includes
#   the original features, their higher-degree polynomial terms, and interaction terms.
#   For example
#       if X originally has one feature [x], after transformation:
#           X_poly will include [1, x, x^2].
#       If X has two features [x1, x2], then X_poly will include:
#           [1, x1, x2, x1^2, x1*x2, x2^2].
# At this stage:
# - The `fit_transform` method calculates the necessary polynomial terms for the data.
# - The resulting X_poly has an additional column for each new polynomial and interaction term.
# - The first column in X_poly is typically a column of ones (for the intercept term in regression models).
X_poly = regressor_poly.fit_transform(X)
# print(X_poly)
# This will be used to fit the transformed polynomial features (X_poly) to the target variable (y).
regressor_slr2 = LinearRegression()

# Fit the linear regression model to the transformed feature matrix (X_poly) and target variable (y).
#   X_poly includes the original features, polynomial terms (e.g., x^2), and interaction terms (e.g., x1*x2).
# - The LinearRegression model treats X_poly as a set of linear predictors, regardless of their nonlinear nature in the original space.
# - This step computes the optimal coefficients (weights) for each term in X_poly to minimize the error between the predictions and actual target values (y).
regressor_slr2.fit(X_poly, y)
# After fitting:
# - The model has learned the relationship between the polynomial-transformed features and the target variable.
# - These learned coefficients can be used to make predictions on new data (after applying the same polynomial transformation).

print('\nStep 5 - Visualising the Linear Regression results')
plt.scatter(X, y, color='red')
plt.plot(X, regressor_slr.predict(X), color='blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

print('\nStep 6 - Visualising the Polynomial Regression results')
plt.scatter(X, y, color='red')
plt.plot(X, regressor_slr2.predict(X_poly), color='blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


print('\nStep 7 - Visualising the Polynomial Regression results (for higher resolution and smoother curve)')
regressor_poly_degree4 = PolynomialFeatures(degree=4)
X_poly2 = regressor_poly_degree4.fit_transform(X)
regressor_slr3 = LinearRegression()
regressor_slr3.fit(X_poly2, y)

# Generate a finer grid of X values for a smoother curve
X_grid = np.arange(min(X.ravel()), max(X.ravel()), 0.01)  # Finer step for smoother curve
X_grid = X_grid.reshape((len(X_grid), 1))  # Reshape to 2D for compatibility

X_grid_poly = regressor_poly_degree4.transform(X_grid)

plt.scatter(X, y, color='red')  # Original data points
plt.plot(X_grid, regressor_slr3.predict(X_grid_poly), color='blue')  # Smoother regression curve
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

print('\nStep 7 - Predicting a new result with Linear Regression')
print(regressor_slr.predict([[6.5]]))

print('\nStep 8 - Predicting a new result with Polynomial Regression')
print(regressor_slr2.predict(regressor_poly.transform([[6.5]])))
print(regressor_slr3.predict(regressor_poly_degree4.transform([[6.5]])))