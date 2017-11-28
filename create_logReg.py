import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import KFold, cross_val_score


# Start Logistic Regression
dataset = pd.read_csv('CSV Dataset_LogReg_Followers_Nov23.csv')
print(dataset)

# prepare datasets to be fed in the regression model
# predict stock movement class given all stated EV's
CV = dataset.stock_mvmt.values.reshape((len(dataset.stock_mvmt), 1))
data = (dataset.ix[:,'avjSubj_std':'numVerified_over_revenue_std'].values).reshape((len(dataset.stock_mvmt), 8))

# Create a KNN object
LogReg = LogisticRegression()

# Train the model using the training sets
LogReg.fit(data, CV.ravel())

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
#kf = KFold(len(CV), n_splits=5)

kf = KFold(n_splits=10, shuffle=True, random_state=None)
scores = cross_val_score(model, data, CV.ravel(), cv=kf)


print("Accuracy of every fold in 5 fold cross validation: ", abs(scores))
print("Mean of the 5 fold cross-validation: %0.2f" % abs(scores.mean()))
