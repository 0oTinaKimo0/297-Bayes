import numpy as np 
import pandas as pd 
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Load the adult income data from the csv file using pandas
data = pd.read_csv('adult.csv', header=None)
data.columns = ['age', 'work', 'fnlwgt', 'edu', 'edunum', 'marital', 'occu', 'relation', 'race', 'sex', 'gain', 'loss', 'hpw', 'country', 'income']
features_raw = ['age', 'work', 'fnlwgt', 'edunum', 'marital', 'occu', 'relation', 'race', 'sex', 'gain', 'loss', 'hpw', 'country']
features = ['age', 'fnlwgt', 'edunum', 'marital', 'relation', 'sex', 'gain', 'loss', 'hpw', 'country']
data.replace(' ?', np.NaN, inplace=True)
data.dropna(inplace=True)

# Data preprocessing
data['income'].replace(' <=50K', 0, inplace=True)
data['income'].replace(' >50K', 1, inplace=True)
data['sex'].replace(' Female', 0, inplace=True)
data['sex'].replace(' Male', 1, inplace=True)
data['country'].replace(' United-States', 1, inplace=True)
data['country'].where(data['country'] == 1, 0, inplace=True)
data['marital'].replace(dict.fromkeys([' Married-AF-spouse', ' Married-civ-spouse', ' Married-spouse-absent'], 1), inplace=True)
data['marital'].where(data['marital'] == 1, 0, inplace=True)
data['relation'].replace(' Own-child', 1, inplace=True)
data['relation'].where(data['relation'] == 1, 0, inplace=True)

Y = data['income']
X = data[features]
print('Using features: ' + str(features))

# Define the Naive Bayes model - Gaussian
naiveBayesModel = GaussianNB()

# Split training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)

# Test options and evaluation metric
scoring = 'accuracy'

# Fit the training set
naiveBayesModel.fit(X_train, Y_train)

# Predict for the test set
pred_naiveBayesModel = naiveBayesModel.predict(X_test)

# Get metrics for this specific split of train-test data
print(classification_report(Y_test, pred_naiveBayesModel))
print(confusion_matrix(Y_test, pred_naiveBayesModel))

# Model Performance
# Set performance parameters
kfold = model_selection.KFold(n_splits=10)

# Call the cross validation function
cv_results = model_selection.cross_val_score(GaussianNB(), X_train, Y_train, cv=kfold, scoring=scoring)

# Display the mean and standard deviation of the prediction
print("%s: %f %s: (%f)" % ('Naive Bayes accuracy', cv_results.mean(), '\nNaive Bayes StdDev', cv_results.std()))


