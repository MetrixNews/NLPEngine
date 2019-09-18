# https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html --> scikit-learn tutorial to train multiple models simultaneuously for experimentation
# https://www.kaggle.com/jeffd23/10-classifier-showdown-in-scikit-learn --> kaggle tutorial to train multiple models simultaneously for experimentation
# https://scikit-learn.org/stable/modules/multiclass.html --> scikit-learn list of viable multiclass classifiers (we want "inherently multiclass" or "one vs all")
# https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html --> outlines the "one vs all" fun ction in more detail, but the link above also includes the code

#"One vs All" method says that each article can only be assigned to one bias class. In order to do this, the sklearn.multiclass.OneVsRestClassifier function will create five bias models. the first model
# predicts liberal articles, the second predicts center-liberal, the third center, the fourth center conservative, and the fifth conservative. For example, the liberal model will look at everything that
# makes center articles different than all the rest and return 1 if a test article is center or 0 if it is anything else. It loops through the five models where each picks out articles that belong to that
# bias label 

# AJ's Note --> Objective is to train, test, and plot several multiclass classification models to figure out which one works best with our data to predict bias of articles. I am most familiar with
# accuracy, precision, recall, and F1 measures to determine model performance. So can we plot all those for each model?

#MPC TODO
# Logging

# Data processing / visualization]
import math
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize 
from sklearn.feature_selection import RFECV
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn import preprocessing

import logging
import os.path
import logging.config
import gc

#Classification Models
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, RidgeClassifier

#Performance measurements
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

#Import Data & Preparation
def main():
   csv.field_size_limit(100000000)

   news_corpus = []

   logging.info("Inserting articles into news_corpus[]")
   # with open('data/articles.csv', 'r') as f:
      
   #    reader = csv.reader((line.replace('\0', '') for line in f), delimiter=',')
      
   #    numcols = len(next(reader))
   #    f.seek(0)

   #    next(reader)
   #    next(reader)

   #    for row in reader:
   #       try:
   #          article = []
   #          for i in range(0, numcols):
   #             article.append(row[i])
   #          news_corpus.append(article)
   #       except Exception as e:
   #          print(e)
   #    del article

   df_chunk = pd.read_csv(r'data/articles.csv', chunksize=5000)

   logging.info("Garbage Collection...")
   gc.collect()

  #Cross Validation Data Partition
   X_arr = []
   Y_arr = []
   # logging.info("Partitioning news_corpus into X and Y arrays")
   # for article in news_corpus:     
   #    tmp_x = []
   #    for i in range(0, numcols):
   #       if i == 19:
   #          continue
   #       else:
   #          tmp_x.append(article[i])
   #    X.append(tmp_x)
   #    Y.append([article[19]])
   
   # del tmp_x
   # del news_corpus

   for chunk in df_chunk:
      tmp_x = []
      for i in range(0, numcols):
         if i == 19:
            continue
         else:
            tmp_x.append(chunk[i])
      X_arr.append(tmp_x)
      Y.append([chunk[19]])

   logging.info("Garbage Collection...")
   gc.collect()

   logging.info("Converting X to DataFrame")

   # logging.info("Converting X to dictionary")
   # X_dict = X.to_dict(orient='records') 

   # logging.info("Creating DictVectorizer Spase=False")
   # dv_X = DictVectorizer(sparse=False)

   # logging.info("Performing DictVectorization on X using X_dict")
   # X = dv_X.fit_transform(X_dict)

   logging.info("Performing get_dummies on X. dummy_na=True, sparse=True")
   X_dummy = pd.get_dummies(X_arr, dummy_na=True, sparse=True)

   logging.info("Convering X to numpy array")
   X = np.array(X_dummy)
   logging.info("Convering Y to numpy array")
   Y = np.array(Y_arr)

   logging.info(f"X shape {X.shape}")
   logging.info(f"Y Shape {Y.shape}")

   logging.info("Generating x_train, x_test, y_train, y_test. test_size=0.22, random_state=42")
   x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.22, random_state=42)

   del X
   del Y
   logging.info("Garbage Collection...")
   gc.collect()

      #OneVsRestClassifier(DecisionTreeClassifier()),
      #OneVsRestClassifier(RandomForestClassifier()),
      #OneVsRestClassifier(GaussianNB()),
      #OneVsRestClassifier(BernoulliNB()),

   classifiers = [   

      OneVsRestClassifier(AdaBoostClassifier()),
      OneVsRestClassifier(GradientBoostingClassifier())
      
   ]

   logging.info("Beginning Classification")
   for clf in classifiers:
      try:
         name = clf.estimator.__class__.__name__

         clf.fit(x_train, y_train)

         logging.info("="*30)
         
         logging.info('****Results****')
         print()
         logging.info(name)
         logging.info('***************')

         # pipe = PipelineREF([
         #    ('stf_scaler', preprocessing.StandardScaler()),
         #    ('ET', clf)
         # ])
         # rfe = RFECV(clf, step=50, min_features_to_select=300)
         # rfe.fit(x_train, y_train)

         #feature_selector_cv = RFECV(pipe, step=50, min_features_to_select=300)
         #feature_selector_cv.fit(x_test, Y)
         
         train_predictions = clf.predict(x_test)
         #train_predictions = feature_selector_cv.predict(x_test)

         f1 = f1_score(y_test, train_predictions, labels=None, pos_label=1, average='micro', sample_weight=None)
         logging.info("F1: {:.4%}\n".format(f1))
      except Exception as e:
         print(e)
         continue

   logging.info("="*30)

if __name__ == "__main__":
   logging.basicConfig(level=logging.INFO, 
   format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
   handlers=[
      logging.FileHandler("logs/ml.log"),
      logging.StreamHandler()
   ])

   logger = logging.getLogger()

   logger.info('bias_ml.py started')

   main()

   logging.debug('bias_ml.py Finished')

   logging.shutdown()

class PipelineREF(Pipeline):
   def fit(self, X, y=None, **fit_params):
      super(PipelineREF, self).fit(X, y, **fit_params)
      self.feature_importance_ = self.steps[-1][-1].feature_importance_
      return self


# Once we choose which models to use and refine the chosen models, we can do one of two things: 
# 1) Choose the highest performing model to deploy
# 2) Keep all well-performing models and create a voting system. Ex: 6 models say article is center, 1 says it is liberal, and 1 says it is conservative.
#    Voting system would take the majority vote to label it center 


# Push to deployment