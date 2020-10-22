import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np

data = pd.read_csv('adult.csv', header=None)
data.columns = ['age', 'class', 'fnlwgt', 'edu', 'edunum', 'marital', 'occu', 'relation', 'race', 'sex', 'gain', 'loss', 'hpw', 'country', 'income']
features = ['age', 'class', 'fnlwgt', 'edunum', 'marital', 'occu', 'relation', 'race', 'sex', 'gain', 'loss', 'hpw', 'country']
numerical = ['age', 'fnlwgt', 'edunum', 'gain', 'loss', 'hpw']
data.replace(' ?', np.NaN, inplace=True)
num_rows = len(data.index)
print("Number of rows: " + str(num_rows))
data.dropna(inplace=True)
print("Number of valid rows without missing values: " + str(len(data.index)))
print("Percentage dropped: {}%". format(100*(1-len(data.index)/num_rows)))
print('Class labels:', np.unique(data['class']))
print('Class labels:', np.unique(data['marital']))
print('Class labels:', np.unique(data['occu']))
print('Class labels:', np.unique(data['relation']))
print('Class labels:', np.unique(data['race']))
print('Class labels:', np.unique(data['country']))

print(data.head(20))