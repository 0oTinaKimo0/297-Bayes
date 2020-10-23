import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data and define column names
data = pd.read_csv('adult.csv', header=None)
data.columns = ['age', 'work', 'fnlwgt', 'edu', 'edunum', 'marital', 'occu', 'relation', 'race', 'sex', 'gain', 'loss', 'hpw', 'country', 'income']

# Decide whether to drop NA or to perform imputation
data.replace(' ?', np.NaN, inplace=True)
num_rows = len(data.index)
print("Number of rows: " + str(num_rows))
data.dropna(inplace=True)
print("Number of valid rows without missing values: " + str(len(data.index)))
print("Percentage dropped: {}%". format(100*(1-len(data.index)/num_rows)))

# Try feature selection via dimensionality reduction
features = ['age', 'work', 'fnlwgt', 'edunum', 'marital', 'occu', 'relation', 'race', 'sex', 'gain', 'loss', 'hpw', 'country']
numerical = ['age', 'fnlwgt', 'edunum', 'gain', 'loss', 'hpw']
print('workclass - class labels:', np.unique(data['work']))
print('occupation - class labels:', np.unique(data['occu']))
print('marital - class labels:', np.unique(data['marital']))
print('relationship - class labels:', np.unique(data['relation']))
print('race - class labels:', np.unique(data['race']))
print('country - class labels:', np.unique(data['country']))
plt.hist(data['work'])
plt.title('Work classes distribution')
plt.show()
plt.hist(data['occu'])
plt.title('Occupation distribution')
plt.show()
print(data['country'].describe())

# Describe the distribution of the target classes
print(data['income'].describe())
data['income'].replace(' <=50K', 0, inplace=True)
data['income'].replace(' >50K', 1, inplace=True)
