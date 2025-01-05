print('Step 1 - Importing the libraries')

#used for numerical operations and array manipulation
import numpy as np

#used for creating visualizations and plots.
import matplotlib.pyplot as plt

#used for data manipulation and analysis, especially with tabular data
import pandas as pd


print('\nStep 2 - Importing the dataset')
dataset = pd.read_csv('Data.csv')


print('\nStep 3 - Denoting predictors (X) and target (y)')
#iloc - stands locate indexes, [rows, columns]
X = dataset.iloc[:, :-1].values # all rows, all columns except last
y = dataset.iloc[:, -1].values # all rows, only last column
print(X)
print(y)

print('\nStep 4 - Taking care of missing data')
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# - This computes the mean of each column in the selected range (1 to 2)
# - The computed means are stored in the imputer object
imputer.fit(X[:, 1:3])

# - Transform the subset of columns (1 and 2) using the computed means
# - Replaces the missing values in these columns with their respective means
# - The result is a modified version of the subset
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)


print('\nStep 5 - Encoding categorical variables')

print('Step 5.1 - Encoding Independent variables')
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# 'encoder': Name of the transformation
# OneHotEncoder(): Specifies to use OneHotEncoding
# [0]: Indicates the column index (0) to be transformed
# Keeps the remaining columns unchanged (passes them through as they are)
ct = ColumnTransformer(
    transformers = [
        ('encoder', OneHotEncoder(), [0])
    ],
    remainder = 'passthrough'
)

# Apply the ColumnTransformer to the dataset `X`
# Step 1: `fit` - Learns the transformation rules from the specified column(s) in `X`
# Step 2: `transform` - Applies the learned rules to transform the data
# Result: Returns a modified version of `X` where the specified column(s) are transformed,
#         and other columns are either retained as-is or dropped (depending on `remainder` argument)
#fit_transform returns a NumPy array, but in some cases, you may want to ensure the result is explicitly a NumPy array.
X = np.array(ct.fit_transform(X))
print(X)

print('Step 5.2 - Encoding Dependent variables')
#The LabelEncoder class in scikit-learn is used for encoding categorical labels into numerical format.
#For ct.fit_transform(X), np.array() is used because the result is typically a sparse matrix,
#       which is converted into a dense NumPy array for compatibility.
# In contrast, le.fit_transform(y) directly returns a dense NumPy array, so np.array() is not needed.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y)

print('\nStep 6 - Splitting the dataset into the Training set and Test set')
from sklearn.model_selection import train_test_split

#test_size=0.2 - Specifies that 20% of the data will be used for testing, and 80% for training.
#random_state=1 - Ensures reproducibility.
#       By setting a random seed (random_state), the data is split the same way every time you run the code.
#       This ensures consistent results when you test or train the model multiple times.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# print(X_train)
# print("\n")
# print(X_test)
# print("\n")
# print(y_train)
# print("\n")
# print(y_test)

print('\nStep 7 - Feature Scaling')
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#Taking only age and salary columns from X_train
# Apply feature scaling (standardization) on the 4th and 5th columns (index 3 and 4) of X_train
X_train[:, 3:5] = sc.fit_transform(X_train[:, 3:])

#Apply the same scaling transformation to the corresponding columns (4th and 5th)
#       of X_test using the parameters from X_train
#transform is used on X_test to scale the test set using the parameters
#       (mean, standard deviation) derived from X_train, ensuring no data leakage.
X_test[:, 3:5] = sc.transform(X_test[:, 3:])

print("\n")
print(X_train)
print("\n")
print(X_test)