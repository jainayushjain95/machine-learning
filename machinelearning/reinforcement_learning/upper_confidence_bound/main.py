# Step 1 - Importing the necessary libraries
import numpy as np  # For mathematical operations
import matplotlib.pyplot as plt  # For data visualization
import pandas as pd  # For handling datasets
import math  # For using mathematical functions like sqrt and log

print('Step 1 - Importing the libraries')

# Step 2 - Importing the dataset
print('\nStep 2 - Importing the dataset')
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Extract the number of users (rows) and number of ads (columns)
NO_OF_USERS = dataset.shape[0]  # Total number of users
NO_OF_ADS = dataset.shape[1]  # Total number of different ads

# Initialize lists and variables
ads_shown = []  # Track the ads shown to users at each round
number_of_times_shown = [0] * NO_OF_ADS  # Track how many times each ad was shown
sums_of_rewards = [0] * NO_OF_ADS  # Track the sum of rewards for each ad
total_reward = 0  # Track the total rewards obtained

print(f'NO_OF_USERS: {NO_OF_USERS}, NO_OF_ADS: {NO_OF_ADS}')

# Step 3 - Implementing the Upper Confidence Bound (UCB) Algorithm
print('\nStep 3 - Implementing the Upper Confidence Bound')

# Iterate over each user (one round of ad display per user)
for i in range(NO_OF_USERS):
    max_upper_confidence_bound = 0  # Initialize the max UCB value
    ad_to_show = 0  # Variable to store the ad to be shown

    # Iterate over each ad to calculate UCB and choose the best one
    for j in range(NO_OF_ADS):
        if number_of_times_shown[j] > 0:
            # Calculate the average reward for the ad (estimated CTR)
            average_reward = sums_of_rewards[j] / number_of_times_shown[j]

            # Calculate the confidence bound (exploration term) using the formula
            delta_i = math.sqrt(3 / 2 * math.log(i + 1) / number_of_times_shown[j])

            # Calculate the final Upper Confidence Bound
            upper_confidence_bound = average_reward + delta_i
        else:
            # If the ad has never been shown, assign a very large UCB to ensure selection
            # It forces the algorithm to select ads that have never been shown at least once.
            upper_confidence_bound = 1e400

        # Select the ad with the highest UCB
        if upper_confidence_bound > max_upper_confidence_bound:
            max_upper_confidence_bound = upper_confidence_bound
            ad_to_show = j

    # Store the shown ad in the ads_shown list
    ads_shown.append(ad_to_show)
    # Update the number of times the chosen ad was shown
    number_of_times_shown[ad_to_show] += 1
    # Get the reward for the shown ad from the dataset (1 if clicked, 0 if not)
    reward = dataset.values[i, ad_to_show]
    # Update the sum of rewards for the shown ad
    sums_of_rewards[ad_to_show] += reward
    # Update the total reward
    total_reward += reward

print('\nStep 3 - Visualising the results')
plt.hist(ads_shown)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
