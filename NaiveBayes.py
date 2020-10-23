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
data['work'].replace(' Without-pay', ' ?', inplace=True)
data.replace(' ?', np.NaN, inplace=True)
data.dropna(inplace=True)

# Data preprocessing
data['income'].replace(' <=50K', 0, inplace=True)
data['income'].replace(' >50K', 1, inplace=True)
data['sex'].replace(' Female', 0, inplace=True)
data['sex'].replace(' Male', 1, inplace=True)
data['country'].replace(' United-States', 1, inplace=True)
data['country'].where(data['country'] == 1, 0, inplace=True)
data['marital'].replace(dict.fromkeys([' Married-AF-spouse', ' Married-civ-spouse'], 1), inplace=True)
data['marital'].where(data['marital'] == 1, 0, inplace=True)
data['relation'].replace(' Own-child', 1, inplace=True)
data['relation'].where(data['relation'] == 1, 0, inplace=True)
data['work'].replace(dict.fromkeys([' Self-emp-not-inc', ' Self-emp-inc'], ' Self-emp'), inplace=True)
data['work'].replace(dict.fromkeys([' Local-gov', ' State-gov', ' Federal-gov'], ' Government'), inplace=True)
reorder_colnames = ['income', 'age', 'fnlwgt', 'edu', 'edunum', 'marital', 'occu', 'relation', 'race', 'sex', 'gain', 'loss', 'hpw', 'country', 'work']
data = data.reindex(columns=reorder_colnames)
data = pd.get_dummies(data, columns=['work'])
reorder_colnames = ['income', 'age', 'fnlwgt', 'edu', 'edunum', 'marital', 'relation', 'race', 'sex', 'gain', 'loss', 'hpw', 'country',
                    'work_ Private','work_ Self-emp','work_ Government', 'occu']
data = data.reindex(columns=reorder_colnames)
data = pd.get_dummies(data, columns=['occu'])
reorder_colnames = ['income', 'age', 'fnlwgt', 'edu', 'edunum', 'marital', 'relation', 'race', 'sex', 'gain', 'loss', 'hpw', 'country',
                    'work_ Private','work_ Self-emp','work_ Government', 'occu_ Adm-clerical', 'occu_ Armed-Forces', 'occu_ Craft-repair', 'occu_ Exec-managerial', 'occu_ Farming-fishing', 'occu_ Handlers-cleaners',
            'occu_ Machine-op-inspct', 'occu_ Other-service', 'occu_ Priv-house-serv', 'occu_ Prof-specialty', 'occu_ Protective-serv', 'occu_ Sales',
            'occu_ Tech-support', 'occu_ Transport-moving']
data = data.reindex(columns=reorder_colnames)
data = pd.get_dummies(data, columns=['race'])
print(list(data.columns))

features = ['age', 'fnlwgt', 'work_ Private','work_ Self-emp','work_ Government', 'edunum', 'marital', 'relation', 'sex', 'gain', 'loss', 'hpw', 'country',
            'occu_ Adm-clerical', 'occu_ Armed-Forces', 'occu_ Craft-repair', 'occu_ Exec-managerial', 'occu_ Farming-fishing', 'occu_ Handlers-cleaners',
            'occu_ Machine-op-inspct', 'occu_ Other-service', 'occu_ Priv-house-serv', 'occu_ Prof-specialty', 'occu_ Protective-serv', 'occu_ Sales',
            'occu_ Tech-support', 'occu_ Transport-moving', 'race_ Amer-Indian-Eskimo', 'race_ Asian-Pac-Islander', 'race_ Black', 'race_ Other', 'race_ White']
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


