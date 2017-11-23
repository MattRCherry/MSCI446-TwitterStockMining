import pandas as pd
import matplotlib.pyplot as plt
import math
import pylab as P
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

# # read our dataset from csv
# # use 'csv_dataset_revenue_norm.csv' for revenue-normalized data
# # use 'csv_dataset_followers_norm.csv' for Twitter-followers-normalized data
# df = pd.read_csv('CSV Dataset_Revenue_November22.csv', index_col=False)
#
# # isolate the column labels of our explanatory variables
# ev_col_labels = df.columns.values
# ev_col_labels = ev_col_labels[2:10]
#
# # plot all of our numerical EV's against our class variable
# # can comment out if it is not necessary
# for label in ev_col_labels:
#     plt.figure(1)
#     plt.scatter(getattr(df, label), df.change_std, color='blue')
#     plt.title("change_std as a function of " + label)
#     plt.xlabel(label)
#     plt.ylabel("change_std")
#     plt.show()
#
# # isolate our explanatory and class variables
# data = df.as_matrix(columns=ev_col_labels)
# cv = df.change_std
# print(data)
# print(cv)
#
# # fit a linear regression model
# regr = linear_model.LinearRegression()
# regr.fit(data, df.change_std)
#
# # use our model to predict on our overall dataset
# predicted_results = regr.predict(data)
#
# print("\nResults:")
# # The coefficients (m, b) of y = m1x1 + m2x2 + ... + b
# print('Coefficients (m): \n', ev_col_labels, '\n', regr.coef_)
# print('Intercept (b): \n', regr.intercept_)
#
# # The mean square error MSE or the mean residual sum of square of errors
# MSE = mean_squared_error(cv, predicted_results)
# RMSE = math.sqrt(MSE)
# # Explained variance score: 1 is perfect prediction
# R2 = r2_score(cv, predicted_results)
#
# print("Mean residual sum of squares =", MSE)
# print("RMSE =", RMSE)
# print("R2 =", R2)

# Start Logistic Regression
dataset = pd.read_csv('CSV Dataset_LogReg_Followers_Nov22.csv')
print(dataset)

# prepare datasets to be fed in the regression model
# predict stock movement class given all stated EV's
CV = dataset.stock_mvmt.reshape((len(dataset.stock_mvmt), 1))
data = (dataset.ix[:,'avjSubj_std':'numVerified_over_followers_std'].values).reshape((len(dataset.stock_mvmt), 8))

# Create a KNN object
LogReg = LogisticRegression()

# Train the model using the training sets
LogReg.fit(data, CV)

# the model
print('Coefficients (m): \n', LogReg.coef_)
print('Intercept (b): \n', LogReg.intercept_)


#predict the class for each data point
predicted = LogReg.predict(data)
print("Predictions: \n", np.array([predicted]).T)

# predict the probability/likelihood of the prediction
print("Probability of prediction: \n",LogReg.predict_proba(data))

print("Accuracy score for the model: \n", LogReg.score(data,CV))

print(metrics.confusion_matrix(CV, predicted, labels=["UP","DOWN"]))

# Calculating 5 fold cross validation results
model = LogisticRegression()
kf = StratifiedKFold(len(CV), n_folds=5)
scores = cross_val_score(model, data, CV, cv=kf)
print("Accuracy of every fold in 5 fold cross validation: ", abs(scores))
print("Mean of the 5 fold cross-validation: %0.2f" % abs(scores.mean()))
