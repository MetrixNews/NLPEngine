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
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

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

import logging
import os.path
import logging.config
import gc
from slacker_log_handler import SlackerLogHandler, NoStacktraceFormatter

#Import Data & Preparation
def main():
   news_corpus = []

   logging.info("Inserting articles into news_corpus[]")
   news_corpus = pd.read_csv('data/articles_trimmed.csv')

   logging.info(news_corpus)
      # 0 
      # 1 - id
      # 2 - source_name
      # 3 - title X
      # 4 - url X
      # 5 - published_at X
      # 6 - vaderpos
      # 7 - vaderneu
      # 8 - vaderneg
      # 9 - anger
      # 10 - anticipation
      # 11 - disgust
      # 12 - fear
      # 13 - joy
      # 14 - sadness
      # 15 - surprise
      # 16 - trust
      # 17 - topic
      # 18 - biasness
      # 19 - political_bias
      # 20 - topic
      # 21.. - tfidf

   #Cross Validation Data Partition
   logging.info("Splitting data into X and Y")
   X = []
   Y = []
   for index, row in news_corpus.iterrows():
      X.append(row)
      Y.append(row['political_bias'])

   logging.info("Converting X to dataframe")
   X = pd.DataFrame(X)

   X = X.drop(['political_bias'], axis=1)

   logging.info(X)

   logging.info("Preforming get_dummie on X")
   X = pd.get_dummies(X, dtype=float, columns=['topic', 'source_name', 'biasness'])
   
   logging.info(X)

   logging.info("Coverting X to array")
   X = np.array(X)
   logging.info('Converting Y to array')
   Y = np.array(Y)

   logging.info(X.shape)
   logging.info(Y.shape)

   logging.info("Splitting X and Y into x_train, x_test, y_train, y_test. test_size=0.22, random_state=42")
   x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.22, random_state=42, shuffle=True)
   

   # OneVsRestClassifier(DecisionTreeClassifier()),
   # OneVsRestClassifier(RandomForestClassifier()),
   # OneVsRestClassifier(AdaBoostClassifier()),
   # OneVsRestClassifier(GaussianNB()),
   # OneVsRestClassifier(BernoulliNB()),
   # OneVsRestClassifier(LogisticRegression()),
   # OneVsRestClassifier(RidgeClassifier())

   # rfe = RFECV(clf, step=50)
   # rfe.fit(x_train, y_train)

   #gbc = GradientBoostingClassifier()
   clf = GradientBoostingClassifier(learning_rate=0.2, n_estimators=70, min_samples_split=500, min_samples_leaf=50, max_depth=20, max_features='sqrt', subsample=0.8)

   # parameters = {
   #    'n_estimators':
   #    [
   #       50, 75, 100, 250, 500, 1000
   #    ],
   #    'learning_rate':
   #    [
   #       1.0, 0.75, 0.50, 0.25, 0.10, 0.01, 0.001
   #    ],
   #    'min_samples_split':
   #    [
   #       0.01, 0.001, 0.0001
   #    ],
   #    'max_features':
   #    [
   #       'auto', 'log2', 'None'
   #    ]
   # }

   #clf = GridSearchCV(estimator=gbc, param_grid=parameters, cv=StratifiedKFold(10))
   
   name = 'Gradient Boosting Classifier'
   
   rfecv = RFECV(estimator=clf, step=50, cv=StratifiedKFold(5), scoring='accuracy', n_jobs=-1)

   logging.info(f"Running {name} Classifier")

   rfecv.fit(x_train, y_train)

   logging.info('Optimal number of features: {}'.format(rfecv.n_features_))

   logging.info(f'Dropping Features: {np.where(rfecv.support_ == False)[0]}')

   x_train.drop(x_train.columns[np.where(rfecv.support_ == False)[0]], axis=1, inplace=True)

   clf.fit(x_train, y_train)

   train_predictions = clf.predict(x_test)

   f1 = f1_score(y_test, train_predictions, labels=None, pos_label=1, average='micro', sample_weight=None)
   logging.info("Results (Accuracy): {:.4%}".format(f1)) 

   cm = confusion_matrix(y_test, train_predictions, labels=None, sample_weight=None)
   logging.info(f"Confusion Matrix: \n{cm}")

   #plotting
   fig = plt.figure(figsize=(16, 9))
   plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
   plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
   plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
   plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)
   plt.savefig('data/Recursive Feature Elimination With Cross-Validation.png')
   plt.close(fig)

   dset = pd.DataFrame()
   dset['attr'] = x_train.columns
   dset['importance'] = rfecv.estimator_.feature_importances_

   dset = dset.sort_values(by='importance', ascending=False)
   logging.info(dset)

   fig2 = plt.figure(figsize=(16, 14))
   plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
   plt.title('RFECV - Feature Importances', fontsize=20, fontweight='bold', pad=20)
   plt.xlabel('Importance', fontsize=14, labelpad=20)
   plt.savefig('data/RFECV - Feature Importance.png')
   plt.close(fig2)

   logging.info("="*30)

if __name__ == "__main__":
   slack_handler = SlackerLogHandler('xoxp-380126610855-379116498995-764208393409-f350bee66032dac0403daecba5cf787e', 'Logs', stack_trace=True)
   logging.basicConfig(level=logging.INFO, 
   format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
   handlers=[
      logging.FileHandler("logs/ml.log"),
      logging.StreamHandler()
   ])

   logger = logging.getLogger()

   logger.addHandler(slack_handler)

   formatter = NoStacktraceFormatter('%(levelname)s:%(asctime)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
   slack_handler.setFormatter(formatter)

   slack_handler.setLevel(logging.DEBUG)

   logging.info('<@UB53EENV9> and <@UB63QHZ2B>, bias_ml.py started')

   main()

   logging.info('<@UB53EENV9> and <@UB63QHZ2B>, bias_ml.py Finished')

   logging.shutdown()


# Once we choose which models to use and refine the chosen models, we can do one of two things: 
# 1) Choose the highest performing model to deploy
# 2) Keep all well-performing models and create a voting system. Ex: 6 models say article is center, 1 says it is liberal, and 1 says it is conservative.
#    Voting system would take the majority vote to label it center 
