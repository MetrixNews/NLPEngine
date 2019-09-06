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
   bias = []
   numcols = 0

   j = 0
   max_count = 500

   with open('data/articles.csv', 'r') as f:
      
      reader = csv.reader((line.replace('\0', '') for line in f), delimiter=',')
      try:
         numcols = len(next(reader))
         f.seek(0)

         next(reader)
         next(reader)
         for row in reader:
            if j == max_count:
               break

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

            article = []
            for i in range(0, numcols):
               article.append(row[i])

            news_corpus.append(article)

            j = j + 1
            
      except csv.Error as e:
         print(e)

   #Cross Validation Data Partition
   X = []
   Y = []
   for article in news_corpus:     
      tmp_x = []
      for i in range(0, numcols):
         if i == 19:
            continue
         else:
            tmp_x.append(article[i])
      X.append(tmp_x)
      Y.append([article[19]])

   X = pd.DataFrame(X)

   X_dict = X.to_dict(orient='records') 

   dv_X = DictVectorizer(sparse=False)

   X = dv_X.fit_transform(X_dict)

   X = np.array(X)
   Y = np.array(Y)

   X = normalize(X)

   print(X.shape)
   print(Y.shape)

   # kf = KFold(n_splits=3, shuffle=True, random_state=None)
   # #kf.get_n_splits(X)

   # print(kf)  

   # for train_index, test_index in kf.split(X):
   #    print("TRAIN:", train_index, "TEST:", test_index)
   #    X_train, X_test = X[train_index], X[test_index]
   #    y_train, y_test = Y[train_index], Y[test_index]

   x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.22, random_state=42)

   #Classifier Showdown

   #OneVsRestClassifier(KNeighborsClassifier(n_neighbors=10)),
   #OneVsRestClassifier(RadiusNeighborsClassifier()),
   #OneVsRestClassifier(NearestCentroid()),  

   classifiers = [   
      OneVsRestClassifier(LinearSVC(max_iter=4000)),
      OneVsRestClassifier(DecisionTreeClassifier()),
      OneVsRestClassifier(RandomForestClassifier()),
      OneVsRestClassifier(AdaBoostClassifier()),
      OneVsRestClassifier(GradientBoostingClassifier()),
      OneVsRestClassifier(GaussianNB()),
      OneVsRestClassifier(BernoulliNB()),
      OneVsRestClassifier(LogisticRegression()),
      OneVsRestClassifier(RidgeClassifier())
   ]

   #log_cols=["Classifier", "Precision", "Recall", "F1_measure"]
   log_cols=["Classifier", "Method", "Value"]
   log = pd.DataFrame(columns=log_cols)

   for clf in classifiers:
      name = clf.estimator.__class__.__name__

      clf.fit(x_train, y_train)
      rfe = RFECV(clf, step=10)
      rfe.fit(x_train, y_train)

      print("="*30)
      
      print('****Results****')

      train_predictions = clf.predict(x_test)

      # prec = precision_recall_fscore_support(y_test, train_predictions, labels=None, pos_label=1, average='micro', sample_weight=None)
      # print("Precision: {:.4%}".format(prec[0]))
      # log_entry = pd.DataFrame([[name, 'Precision', prec[0]]], columns=log_cols)
      # log = log.append(log_entry)
      
      # recall = recall_score(y_test, train_predictions, labels=None, pos_label=1, average='micro', sample_weight=None)
      # print("Recall: {:.4%}".format(recall))
      # log_entry = pd.DataFrame([[name, 'Recall', recall]], columns=log_cols)
      # log = log.append(log_entry)

      f1 = f1_score(y_test, train_predictions, labels=None, pos_label=1, average='micro', sample_weight=None)
      print("F1: {:.4%}".format(f1))
      log_entry = pd.DataFrame([[name, 'F1', f1]], columns=log_cols)
      log = log.append(log_entry)
      
      # auc = roc_auc_score(y_true, y_score, average='macro', sample_weight=None, max_fpr=None)
      # print("AUC: {:.4%}".format(auc))

      cm = confusion_matrix(y_test, train_predictions, labels=None, sample_weight=None)
      print(cm)

      #log_entry = pd.DataFrame([[name, prec[0], recall, f1]], columns=log_cols)
      #log = log.append(log_entry)

   print("="*30)

   #Visualize Performance Metrics
   #Precision
   sns.set_color_codes("muted")
   sns.catplot(x='Value', y='Classifier', hue="Method", data=log, kind='bar')

   plt.xlabel('Precision %')
   plt.title('Classifier Precision')
   plt.show()

   #Recall
   #sns.set_color_codes("muted")
   #sns.barplot(x='Recall', y='Classifier', data=log, color="g")

   # plt.xlabel('Recall %')
   # plt.title('Classifier Recall')
   # plt.show()

   #F1
   # sns.set_color_codes("muted")
   # sns.barplot(x='F1_measure', y='Classifier', data=log, color="r")

   # plt.xlabel('F1 %')
   # plt.title('Classifier F1')
   # plt.show()

   # #AUC
   # sns.set_color_codes("muted")
   # sns.barplot(x='AUC', y='Classifier', data=log, color="y")

   # plt.xlabel('AUC %')
   # plt.title('Classifier AUC')
   # plt.show()

if __name__ == "__main__":
    main()


# Once we choose which models to use and refine the chosen models, we can do one of two things: 
# 1) Choose the highest performing model to deploy
# 2) Keep all well-performing models and create a voting system. Ex: 6 models say article is center, 1 says it is liberal, and 1 says it is conservative.
#    Voting system would take the majority vote to label it center 


# Push to deployment