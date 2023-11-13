import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import re
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit

# Loading file and looking into the dimensions of data

raw_data = pd.read_csv("SMSSpamCollection.tsv",sep='\t',names=['label','text'])
pd.set_option('display.max_colwidth',100)
raw_data.head()

raw_data.loc[:,'label']=raw_data.label.map({'ham':0,'spam':1})


print(raw_data.shape)
pd.crosstab(raw_data['label'],columns = 'label',normalize=True)

#TEST MODELS
#To determine the best model for spam detection, we can compare the standard metrics 
# -- accuracy, precision, recall, f1 score -- for different models. 

def print_metric(metric,scores):
    test_metric='test_'+metric
    train_metric='train_'+metric
    print("Mean test/train %s: %.3f \u00B1 %.4f / %.3f \u00B1 %.4f" % \
          (metric, scores[test_metric].mean(), scores[test_metric].std(), \
           scores[train_metric].mean(), scores[train_metric].std()))
           
def print_model_metrics(model,n_folds):
    metrics=['accuracy','precision','recall','f1']
    scores = cross_validate(model, raw_data.text, raw_data.label, scoring=metrics, cv=n_folds, return_train_score=True)
    for metric in metrics:
        print_metric(metric,scores)