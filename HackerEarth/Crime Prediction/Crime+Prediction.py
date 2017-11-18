
# coding: utf-8

# ## Loading data

# In[1]:

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

get_ipython().magic('matplotlib inline')


# In[2]:

df = pd.read_csv('train1.csv', index_col=0)
df.head(n=6)


# ## Training data missing value treatment 

# In[3]:

index = df[df['cases_handled'] == "UNK"]['cases_handled'].index
df.loc[index, 'cases_handled'] = 1
df['cases_handled'] = pd.to_numeric(df['cases_handled'])

index = df[df['race'] == "UNK"]['race'].index
df.loc[index, 'race'] = "White"

index = df[df['sex'] == "UNK"]['sex'].index
df.loc[index, 'sex'] = "Male"


# In[4]:

X_train = df.drop(['officer_id', 'officer_initials', 'target'], axis=1)
y_train = df['target']


# ## Testing data missing value treatment

# In[5]:

test = pd.read_csv('test1.csv', index_col=0)


# In[6]:

index = test[test['cases_handled'] == "UNK"]['cases_handled'].index
test.loc[index, 'cases_handled'] = 1
test.loc[45, 'cases_handled'] = 5
test['cases_handled'] = pd.to_numeric(test['cases_handled'])

index = test[test['race'] == "UNK"]['race'].index
test.loc[index, 'race'] = "White"

index = test[test['sex'] == "UNK"]['sex'].index
test.loc[index, 'sex'] = "Male"


# In[7]:

X_test = test.drop(['officer_id', 'officer_initials'], axis=1)


# In[8]:

#LABELS = ['target']
NUMERIC_COLUMNS = ['cases_handled']
CATEGORICAL_COLUMNS = ['race', 'sex', 'investigative_findings']

def combine_text_columns(data_frame, to_drop=NUMERIC_COLUMNS + CATEGORICAL_COLUMNS):
    """ Takes the dataset as read in, drops the non-feature, non-text columns and
        then combines all of the text columns into a single vector that has all of
        the text for a row.
        
        :param data_frame: The data as read in with read_csv 
        :param to_drop: Removes the numeric and target label columns.
    """
    # dropping non-text columns that are in the df
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop, axis=1)
   
    # replacing nans with blanks
    text_data.fillna("", inplace=True)
    
   # joining all of the text items in a row (axis=1) with a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)


# In[9]:

from sklearn.preprocessing import FunctionTransformer

get_text_data = FunctionTransformer(combine_text_columns, validate=False)
get_numeric_data = FunctionTransformer(lambda x: x[NUMERIC_COLUMNS], validate=False)
get_categorical_data = FunctionTransformer(lambda x: x[CATEGORICAL_COLUMNS], validate=False)


# In[10]:

from features.SparseInteractions import SparseInteractions


# In[11]:

from sklearn.feature_selection import chi2, SelectKBest

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import MaxAbsScaler

TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'


# In[12]:

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)


# In[13]:

get_ipython().run_cell_magic('time', '', '\n# ignore deprecation warnings in sklearn\nimport warnings\nwarnings.filterwarnings("ignore")\n\n# setting a reasonable number of features before adding interactions\nchi_k = 15\n\n# creating the pipeline object\npl = Pipeline([\n        (\'union\', FeatureUnion(\n            transformer_list = [\n                (\'numeric_features\', Pipeline([\n                    (\'selector\', get_numeric_data)#,\n                    #(\'imputer\', Imputer())\n                ])),\n                \n                (\'categorical_features\', Pipeline([\n                    (\'selector\', get_categorical_data),\n                    (\'le\', MultiColumnLabelEncoder())\n                ])),\n                \n                (\'text_features\', Pipeline([\n                    (\'selector\', get_text_data),\n                    (\'vectorizer\', HashingVectorizer(token_pattern=TOKENS_ALPHANUMERIC,\n                                                     non_negative=True, norm=None, binary=False,\n                                                     ngram_range=(1, 2))),\n                    (\'dim_red\', SelectKBest(chi2, chi_k))\n                ]))\n             ]\n        )),\n        (\'int\', SparseInteractions(degree=2)),\n        (\'scale\', MaxAbsScaler()),\n        (\'clf\', AdaBoostClassifier(DecisionTreeClassifier(), random_state = 42))\n    ])\n\n# fitting the pipeline to our training data\npl.fit(X_train, y_train.values)\n\n# printing the score of our trained pipeline on our test set\n#print("Logloss score of trained pipeline: ", log_loss_scorer(pl, X_test, y_test.values))\n\n# Computing and printing accuracy\naccuracy = pl.score(X_train, y_train)\nprint("\\nAccuracy on test dataset: ", accuracy)')


# In[14]:

# Making predictions
predictions = pl.predict(X_test)
#prob = pl.predict_proba(X_test)[:,1]
dt = {'target':predictions}
#temp = pd.DataFrame(np.array(predictions).reshape(100128,104))
# Formatting correctly in new DataFrame: prediction_df
prediction_df = pd.DataFrame(data=dt, index=test.officer_id)


# Saving prediction_df to csv called "predictions.csv"
prediction_df.to_csv("predictions12.csv")


# In[ ]:



