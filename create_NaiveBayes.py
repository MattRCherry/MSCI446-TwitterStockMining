import pandas as pd
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB

dataset = pd.read_csv('Final datasets/CSV_Followers_Categorical_Nov27.csv')
print(dataset)

# prepare datasets to be fed into the naive bayes model
#predict stock movement given all given EV's
CV =  dataset.categorical.values.reshape((len(dataset.categorical), 1))
data = (dataset.ix[:,'avjSubj_std':'numVerified_over_followers_std'].values).reshape((len(dataset.categorical), 8))

# '''
# #predict grade letter given extra hours
# CV =  dataset.letter_grade.reshape((len(dataset.attend_class), 1))
# data = dataset.extra_hours.reshape((len(dataset.attend_class), 1))
# '''
#
# '''
# #predict grade letter given extra hours and attend class
# CV =  dataset.letter_grade.reshape((len(dataset.attend_class), 1))
# data = (dataset.ix[:,['extra_hours','attend_class']].values).reshape((len(dataset.attend_class), 2))
# '''

# Create model object
NB = GaussianNB()

# Train the model using the training sets
NB.fit(data, CV)

#Model
print("Probability of the classes: ", NB.class_prior_)
print("Mean of each feature per class:\n", NB.theta_)
print("Variance of each feature per class:\n", NB.sigma_)

#predict the class for each data point
predicted = NB.predict(data)
print("Predictions:\n",np.array([predicted]).T)

# predict the probability/likelihood of the prediction
prob_of_pred = NB.predict_proba(data)
print("Probability of each class for the prediction: \n",prob_of_pred)

print("Accuracy of the model: ",NB.score(data,CV))

print("The confusion matrix:\n", metrics.confusion_matrix(CV, predicted, ['UP','DOWN']))

# Calculating 5 fold cross validation results
model = GaussianNB()
kf = KFold(n_splits=10, shuffle=True, random_state=None)
scores = cross_val_score(model, data, CV.ravel(), cv=kf)

print("MSE of every fold in 5 fold cross validation: ", abs(scores))
print("Mean of the 5 fold cross-validation: %0.2f" % abs(scores.mean()))
