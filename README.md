# Assignment 6: Naive Bayes
### Tina Jin and Virginia Weston

### EDA and Data Preprocessing
Before doing exploratory data analysis, we need to first deal with missing data. In the adult income dataset, the missing values are denoted by ?, so we convert ? to np.NAN in order to use the default dropna function to drop rows containing missing values. Since only ~7% data is dropped, we think that the remaining data is sufficient for this classification task, and therefore doesn't need imputation. 
![](/images/Dropna.png)

Then we look into every feature to decide if we should include it in the model. We first look at the format of each column, and print out columns containing non-numerical values, i.e. every column except ['age', 'fnlwgt', 'edunum', 'gain', 'loss', 'hpw']. Since Naive Bayes relies on the assumption that there are no relationships between features, some features need to be manipulated to eliminate duplicate information or strong relationships between features.

We generate the histograms for work class and occupation. From the histograms, we see that for work class, “private” is dominant, and “without pay” is trivial. Therefore, we combine the classes to include only “private”, “government” and “self-employed”. For occupation, we decide to simply onehot encode the column. 

![](/images/Figure_1.png)

![](/images/Figure_2.png)

![](/images/Work.png)

We think that marital status and family relationship are highly correlated, since, for example, if someone is married, they are either husband or wife. Also, the information that if the spouse is in the military or not is captured in the work class column. Thus in our preprocessing steps we manipulated these columns to only contain whether someone’s married or not, and whether someone has children.

We decide to keep the race and sex information to avoid bias. We transform the race column with onehot encoding and the sex column with binary classes 0 for female and 1 for male. We print out the frequencies of the country column, and see that “United States” has dominating frequency. We decide to transform this column with binary classes 0 for immigrant and 1 for United States native.

![](/images/Country.png)

Finally for the EDA, we describe the distribution of the target variable. We see that there are far more instances of income <=50k than income >50k, so this imbalance could impact our model’s precision. One thing we do to alleviate this effect is to stratify the data by y during train-test split, so the target classes distribution is consistent between training and testing data.

![](/images/Income.png)

### Naive Bayes Models
For our Naive Bayes model, we found it valuable to implement Gaussian, Bernoulli, Multinominal, and Complement models to compare accuracy and precision. After fitting the training data, we predicted the x_test values for each model. With these predictions we calculated the accuracy and precision for each of the four models and presented them in the plot seen below. 

![](/images/plot.png)

![](/images/results.png)

We originally split the data 80/20, but then switched to stratifying the data by y so that the target classes would be consistently distributed. Because of this switch, our accuracy and precision for the Gaussian model increased, making it the best distrubution to use on our Naive Bayes model. Although the Bernoulli model had the highest accuracy of 79.6%, it had a low 51.8% precision. The Gaussian model resulted in 78.8% accuracy with 66.2% precision. Because of the low precision in the Bernoulli model we decided that the Gaussian model would be best to use for this data. 
