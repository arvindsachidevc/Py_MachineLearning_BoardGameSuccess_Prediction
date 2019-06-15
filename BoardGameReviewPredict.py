"""

# -*- coding: utf-8 -*-
Created on          06 Jun 2019 at 1:44 PM
Author:             Arvind Sachidev Chilkoor  
Created using:      PyCharm
Name of Project:    Predicting the outcome of a Board Game's success based on specific characteristics

Description:  
This python script reviews data from board games reviews assembled from customers and users and predicts the outcomes,
based on characteristics such as number of players, average rating, playing time, complexity etc.

The prediction models used will be Linear Regression and Random Forest.


"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

# Reading/Load the file/data i.e. the games dataset

games = pd.read_csv('games.csv')

# Print the names of the columns in file.

print(games.columns)
print(games.shape)
print(games.size)

# Make a Histogram of all the ratings in the average_rating column
plt.hist(games["average_rating"])

# Show the plot
plt.show()

# Print the first row of all the games with zero scores
# using the .iloc method on dataframes allows us to index by position
print(games[games["average_rating"] == 0].iloc[0])


# Print the first row of all the games with scores greater than 0
print(games[games["average_rating"] > 0].iloc[0])

# Data pre-processing step: Removing all rows without user reviews.
games = games[games["users_rated"] > 0]

# Removing all rows with missing values
games = games.dropna(axis = 0)

# Make a histogram of all the ratings in the average_rating column
plt.hist(games["average_rating"])

# Show the plot
plt.show()

# Creating the correlation matrix using Seaborn Library which is the statistical data visualization library
# The correlation matrix will plot the games characteristics on X & Y Axis, to tell which are co-related to each other.
# White square indicates that maximum correlation and black indicates zero or no correlation
# using corr() from Pandas
corr_matrix = games.corr()
fig = plt.figure(figsize = (12,12))

# Here we are passing the column labels to X and Y for re-orienting them, else, they will be crowded and unreadable.
yticks = corr_matrix.columns
xticks = corr_matrix.columns

# Initializing K for Heatmap
k = sns.heatmap(corr_matrix, vmax = .8, square = True, yticklabels = yticks, xticklabels= xticks)

# Rotating the labels to the desired orientation and font size
k.set_yticklabels(k.get_yticklabels(), rotation = 0, fontsize = 8)
k.set_xticklabels(k.get_xticklabels(), rotation = 90, fontsize = 8)

# Show the matrix
plt.show()

# Get all the columns from the dataframe
column_list = games.columns.tolist()

# Here we are filtering the columns to remove data we do not want for the moment
# Creating a For Loop to run through the list
column_list = [c for c in column_list if c not in ["bayes_average_rating","average_rating","type","name","id"] ]


# Creating variable to store the prediction
target_predict = "average_rating"


# Generate the training and test datasets

# Generate the training sets
# frac = 0.8 allows us to use 80% of dataset for Training and remaining 20% for Testing
train = games.sample(frac = 0.8, random_state = 1)

# Select all items that are not in the training set, and adding to test set.
# (~ i.e. Binary One's Compliment Operator) gives, those indexes that is not in the training set.
test = games.loc[~games.index.isin(train.index)]

# Print Shapes
print(train.shape)
print(test.shape)

# Intializing the model class

lr = LinearRegression()

# To fit the model with the training data
lr_fit = lr.fit(train[column_list], train[target_predict])


# Generate the predictions for the test set
predicts = lr.predict(test[column_list])

# Compute the errors between our test predictions and actual values
er = mean_squared_error(predicts,test[target_predict])

print("\n**********************************")
print("PREDICTIONS MADE USING THE LINEAR REGRESSION MODEL")
print("\n--------------------------------")
print("PREDICTION :  ", predicts)
print("\n--------------------------------")
print("ERROR - (MEAN SQUARED ERROR) :  ", er)
print("\n--------------------------------")


# Predicting the results using the RANDOM FOREST MODEL - since result is non-linear and has an error

# Initializing the model class for Random Forest
rfr = RandomForestRegressor(n_estimators= 100, min_samples_leaf= 10, random_state= 1)

# Fit the data
rfr.fit(train[column_list], train[target_predict])

# Making the predictions
rfr_predictions = rfr.predict(test[column_list])

# Computing the error between the test predictions and actual values
rfr_err = mean_squared_error(rfr_predictions, test[target_predict])

print("\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print("PREDICTIONS MADE USING THE RANDOM FOREST REGRESSION MODEL")
print("\n--------------------------------")
print("PREDICTION :  ", rfr_predictions)
print("\n--------------------------------")
print("ERROR - (MEAN SQUARED ERROR) :  ", rfr_err)
print("\n--------------------------------")