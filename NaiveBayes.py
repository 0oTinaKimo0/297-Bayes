import numpy as np 
import pandas as pd 
import sklearn.naive_bayes as nb
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt

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

features = ['age', 'fnlwgt', 'work_ Private','work_ Self-emp','work_ Government', 'edunum', 'marital', 'relation', 'sex', 'gain', 'loss', 'hpw', 'country',
            'occu_ Adm-clerical', 'occu_ Armed-Forces', 'occu_ Craft-repair', 'occu_ Exec-managerial', 'occu_ Farming-fishing', 'occu_ Handlers-cleaners',
            'occu_ Machine-op-inspct', 'occu_ Other-service', 'occu_ Priv-house-serv', 'occu_ Prof-specialty', 'occu_ Protective-serv', 'occu_ Sales',
            'occu_ Tech-support', 'occu_ Transport-moving', 'race_ Amer-Indian-Eskimo', 'race_ Asian-Pac-Islander', 'race_ Black', 'race_ Other', 'race_ White']
y = data['income']
X = data[features]
print('Using features: ' + str(features))

# Define the Naive Bayes models
gaussianModel = nb.GaussianNB()
bernoulliModel = nb.BernoulliNB()
multinomialModel = nb.MultinomialNB()
complementModel = nb.ComplementNB()

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Test options and evaluation metric
scoring = 'accuracy'

# Fit the training sets
gaussianModel.fit(X_train, y_train)
bernoulliModel.fit(X_train, y_train)
multinomialModel.fit(X_train, y_train)
complementModel.fit(X_train, y_train)

# Predict for the test sets
predG = gaussianModel.predict(X_test)
predB = bernoulliModel.predict(X_test)
predM = multinomialModel.predict(X_test)
predC = complementModel.predict(X_test)

# Get metrics for this specific split of train-test data
accuracies = [accuracy_score(y_test, predG),
              accuracy_score(y_test, predB),
              accuracy_score(y_test, predB),
              accuracy_score(y_test, predC)]

precisions = [precision_score(y_test, predG),
              precision_score(y_test, predB),
              precision_score(y_test, predM),
              precision_score(y_test, predC)]

models = ["Gaussian",
          "Bernoulli",
          "Multinominal",
          "Complement"]

fig, axs = plt.subplots(ncols = 1, nrows = 2)
axs[0].bar(models, accuracies)
axs[0].set_ylabel("Accuracies")
axs[1].bar(models, precisions)
axs[1].set_ylabel("Precisions")
axs[1].set_xlabel("Model")
plt.show()

print("Gaussian Model accuracy: " + str(accuracy_score(y_test, predG)))
print("Gaussian Model precision: " + str(precision_score(y_test, predG)))
print("Bernoulli Model accuracy: " + str(accuracy_score(y_test, predB)))
print("Bernoulli Model precision: " + str(precision_score(y_test, predB)))
print("Multinominal Model accuracy: " + str(accuracy_score(y_test,predM)))
print("Multinominal Model precision: " + str(precision_score(y_test, predM)))
print("Complement Model accuracy: " + str(accuracy_score(y_test, predC)))
print("Complement Model precision: " + str(precision_score(y_test, predC)))

# Model Performance
# Set performance parameters
kfold = model_selection.KFold(n_splits=10)

# Call the cross validation function
cv_results = model_selection.cross_val_score(nb.GaussianNB(), X_train, y_train, cv=kfold, scoring=scoring)

# Display the mean and standard deviation of the prediction
print("%s: %f %s: (%f)" % ('Naive Bayes accuracy - cross validation', cv_results.mean(), '\nNaive Bayes StdDev - cross validation', cv_results.std()))


