import numpy as np 
import pandas as pd 
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Loading the prostate cancer dataset from the csv file using pandas
data = pd.read_csv('adult.csv', header=None)
data.columns = ['age', 'class', 'fnlwgt', 'edu', 'edunum', 'marital', 'occu', 'relation', 'race', 'sex', 'gain', 'loss', 'hpw', 'country', 'income']
features = ['age', 'class', 'fnlwgt', 'edunum', 'marital', 'occu', 'relation', 'race', 'sex', 'gain', 'loss', 'hpw', 'country']
numerical = ['age', 'fnlwgt', 'edunum', 'gain', 'loss', 'hpw']
# print('Class labels:', np.unique(data['country']))
data.replace(' ?', np.NaN, inplace=True)
data.dropna(inplace=True)
data['income'].replace(' <=50K', 0, inplace=True)
data['income'].replace(' >50K', 1, inplace=True)
Y = data['income']
X = data[numerical]

naiveBayesModel = GaussianNB()

# Split training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# Test options and evaluation metric
scoring = 'accuracy'

#Fitting the training set
naiveBayesModel.fit(X_train, Y_train)

#Predicting for the Test Set
pred_naiveBayesModel = naiveBayesModel.predict(X_test)

#Prediction Probability
prob_pos_naiveBayesModel = naiveBayesModel.predict_proba(X_test)[:, 1]

#Model Performance
#setting performance parameters
kfold = model_selection.KFold(n_splits=10)

#calling the cross validation function
cv_results = model_selection.cross_val_score(GaussianNB(), X_train, Y_train, cv=kfold, scoring=scoring)

#displaying the mean and standard deviation of the prediction
print("%s: %f %s: (%f)" % ('Naive Bayes accuracy', cv_results.mean(), '\nNaive Bayes StdDev', cv_results.std()))

