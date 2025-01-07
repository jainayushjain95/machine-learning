print('Step 1 - Importing the libraries')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('\nStep 2 - Importing the dataset')
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print('\nStep 3 - Splitting the dataset into the Training set and Test set')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print('\nStep 4 - Training the Simple Linear Regression model on the Training set')
#Train (fit) the Linear Regression model on the training data
# X_train: Independent variable(s) (e.g., years of experience)
# y_train: Dependent variable (e.g., salary)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print('\nStep 5 - Predicting the Test set results')
# Use the trained model to predict salaries for the test set (X_test)
y_predict = regressor.predict(X_test)

print('\nStep 6 - Visualising the Training set results')
# Create a scatter plot of actual training data points (red dots)
# Plots the actual training data points in red.
# Xtrain values are on the x-axis (years of experience), and
#       𝑦train values are on the y-axis (salaries).
# plt.scatter(X_train, y_train, color='red')

# Plot the regression line (blue line) using predicted values from the model
# The regression line represents the best-fit line learned by the model
# Plots the regression line (blue line) calculated by the model.
# The line represents the predicted salary for any given year of experience in the training data.
# Arguments Breakdown
#       X_train:
#           The independent variable(s) from the training dataset.
#           Specifies the 𝑥-coordinates (horizontal axis) for the regression line.
#       regressor.predict(X_train)
#           Predicts the dependent variable (𝑦)
#           for each 𝑥-value in X_train using the trained model.
#           Specifies the 𝑦-coordinates (vertical axis) for the regression line
# plt.plot(X_train, regressor.predict(X_train), color='blue')
#
# #Add plot title and labels for better understanding
# plt.title('Salary vs Experience (Training set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')

# Display the plot
# plt.show()

print('\nStep 7 - Visualising the Test set results')
plt.scatter(X_test, y_test, color='red')

# Equivalent to plt.plot(X_train, regressor.predict(X_train), color='blue')
#       As the blue regression line will be same
plt.plot(X_test, y_predict, color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()