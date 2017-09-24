
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
from __future__ import division
from __future__ import print_function

# ignore deprecation warnings in sklearn
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[2]:

df = pd.read_csv('train.csv', encoding='latin-1', index_col=0)

print(df.shape)


# In[3]:

df.info()


# In[4]:

df.head()


# In[5]:

df = df.dropna()


# ## Data Splitting

# In[6]:

X_train = df['tweet']
y_train = df['label']


# In[7]:

#from collections import Counter
#from imblearn.over_sampling import SMOTE

#print('Original dataset shape {}'.format(Counter(y_train)))
#sm = SMOTE(random_state=42)
#X_res, y_res = sm.fit_sample(X_train, y_train)
#print('Resampled dataset shape {}'.format(Counter(y_res)))


# In[11]:

from sklearn.feature_selection import chi2, SelectKBest

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import Imputer
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.preprocessing import MaxAbsScaler

TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'


# In[12]:

from features.SparseInteractions import SparseInteractions


# In[13]:

#%%time

# setting a reasonable number of features before adding interactions
chi_k = 300

# creating the pipeline object
pl = Pipeline([ ('text_features', Pipeline([
                    ('vectorizer', HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC,
                                                     non_negative=True, norm=None, binary=False,
                                                     ngram_range=(1, 3))),
                    ('tfidf', TfidfTransformer()),
                    ('dim_red', SelectKBest(chi2, chi_k))
              ])),
        ('int', SparseInteractions(degree=2)),
        ('scale', MaxAbsScaler()),
        ('clf', AdaBoostClassifier(DecisionTreeClassifier()))
    ])

# fitting the pipeline to our training data
pl.fit(X_train, y_train)

# printing the score of our trained pipeline on our test set
#print("Logloss score of trained pipeline: ", log_loss_scorer(pl, X_test, y_test.values))

# Computing and printing accuracy
#accuracy = pl.score(X_train, y_train)
#print("\nAccuracy on test dataset: ", accuracy)


# In[14]:

accuracy = pl.score(X_train, y_train)
print("\nAccuracy on test dataset: ", accuracy)


# In[15]:

from sklearn.metrics import classification_report

y_pred = pl.predict(X_train)
print(classification_report(y_train, y_pred))


# ## Test Accuracy

# In[19]:

test = pd.read_csv('test.csv', encoding='latin-1', index_col=0)

print(test.shape)


# In[20]:

X_test = test['tweet']


# In[21]:

# Making predictions
predictions = pl.predict(X_test)

#temp = pd.DataFrame(np.array(predictions).reshape(100128,104))
# Formatting correctly in new DataFrame: prediction_df
prediction_df = pd.DataFrame(index=test.id, data=predictions)


# Saving prediction_df to csv called "predictions.csv"
prediction_df.to_csv("predictions.csv")


# In[ ]:



