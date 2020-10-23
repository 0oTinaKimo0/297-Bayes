# Assignment 6: Naive Bayes
### Tina Jin and Virginia Weston

### EDA
Before doing exploratory data analysis, we need to first deal with missing data. In the adult income dataset, the missing values are denoted by ' ?', so we convert ' ?' to np.NAN in order to use the default dropna function to drop rows containing missing values. Since only ~7% data is dropped, we think that the remaining data is sufficient for this classification task, and therefore doesn't need imputation. 
![](/images/Dropna.png)

Then we look into every feature to decide if we should include it in the model. We first look at the format of each column, and print out columns containing non-numerical values, i.e. every column except ['age', 'fnlwgt', 'edunum', 'gain', 'loss', 'hpw'].
Since Naive Bayes relies on the assumption that there are no relationships between features, some features need to be manipulated to eliminate duplicate information or strong relationships between features.
### Data Preprocessing
### Naive Bayes Models
