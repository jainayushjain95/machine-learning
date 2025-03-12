print('Step 1 - Importing the libraries')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('\nStep 2 - Importing the dataset')
#header=None, forcing panda not to consider first row as column names
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transactions = []
num_columns = dataset.shape[1]  # Get the total number of columns

for i in range(len(dataset)):  # Loop through each row
    transactions.append([
        str(dataset.values[i, j]) for j in range(num_columns)
        # Convert each item in the row to a string and store it in a list
    ])
#dataset.values[i, j]: Accesses the element at row index i and column index j.
#str(dataset.values[i, j]): Converts the value into a string (ensures consistent data type).
#range(num_columns): Loops through all column indices from 0 to the last column.
#j: Represents the current column index within row i.

print('\nStep 3 - Training the Apriori Model on the dataset')
from apyori import apriori
rules = apriori(
    transactions = transactions,
    min_support = 0.003,
    min_confidence = 0.2,
    min_lift = 3,
    min_length = 2,
    max_length = 2,
)
# Support measures how frequently an itemset appears in transactions.
# A value of 0.003 means that an item must appear in at least 0.3% of all transactions to be considered.
# Confidence measures how often rule X → Y is correct when X occurs.
# A value of 0.2 means that at least 20% of transactions containing X must also contain Y.
# Lift measures how much stronger the rule is compared to random chance.
# A lift of 3 means the rule is at least 3 times more likely to happen than random co-occurrence.
# Ensures that the generated itemsets contain at least 2 items.
# Ensures that the generated itemsets contain at most 2 items.
# Here, the Apriori model will only generate association rules that involve pairs of items.
# For example, it might find rules like:
# {"Milk"} → {"Bread"}
# {"Butter"} → {"Eggs"}
# But it won’t consider single items (min_length=2) or sets with more than 2 items (max_length=2).

print('\nStep 4 - Displaying the first results coming directly from the output of the apriori function')
results = list(rules)
# print(results)

print('\nStep 5 - Putting the results well organised into a Pandas DataFrame')
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))

resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

print('\nStep 6 - Displaying the results non sorted')
print(resultsinDataFrame)

print('\nStep 7 - Displaying the results sorted by descending lifts')
resultsinDataFrame = resultsinDataFrame.nlargest(n = 10, columns = 'Lift')
print(resultsinDataFrame)