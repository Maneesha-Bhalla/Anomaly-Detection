#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy
import sklearn

print('Python: {}'.format(sys.version))
print('Numpy: {}'.format(numpy.__version__))
print('Pandas: {}'.format(pandas.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('seaborn: {}'.format(seaborn.__version__))
print('scipy: {}'.format(scipy.__version__))
print('sklearn: {}'.format(sklearn.__version__))


# In[21]:


# import the neccessary packakes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[22]:


# load teh dataset from csv
data=pd.read_csv(r"C:\Users\Maneesha-Bhalla\Desktop\Credit Card Fraud Detection\creditcard\creditcard.csv")


# In[23]:


data.shape


# In[24]:


data.describe()


# In[25]:


data=data.sample(frac=0.1,random_state=1)
print(data.shape)


# In[26]:


data.hist(figsize=(30,30))
plt.show()


# In[27]:


Fraud=data[data['Class']==1]
Valid=data[data['Class']==0]
outlier_fraction=len(Fraud)/float(len(Valid))
print(outlier_fraction)
print(len(Fraud))
print(len(Valid))


# In[28]:


cormat=data.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(cormat,vmax=0.8,square=True)
plt.show()


# In[29]:


## get all columns to a list
columns = data.columns.tolist()
columns = [c for c in columns if c not in ["Class"]]
target = "Class"
X = data[columns]
Y = data[target]

print(X.shape)
print(Y.shape)


# In[31]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest 
from sklearn.neighbors import LocalOutlierFactor

state=1

classifiers = {"Isolation Forest": IsolationForest(max_samples=len(X), contamination=outlier_fraction, random_state=state),
            
              "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20, contamination=outlier_fraction)
              
              }


# In[38]:


## fit the model
n_outliers=len(Fraud)

for i, (clf_name, clf) in enumerate (classifiers.items()):
    ## fit the data and tag outliers
    if clf_name == "Local Outlier Factor":
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        y_pred = clf.predict(X)
        scores_pred = clf.decision_function(X)
            
    ## Reshape the prediction values to 0 for valid adn 1 for fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1
    
    n_errors = (y_pred != Y).sum()
    
    ## Run classification metrics
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y,y_pred))
    print(classification_report(Y,y_pred))


# In[ ]:




