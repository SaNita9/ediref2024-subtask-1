import pandas as pd
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pandas as pd
from main import df_train
from main import df_ver
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
import matplotlib
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn import metrics

from sklearn.preprocessing import LabelEncoder
features = ['anger', 'surprise', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'positive', 'negative', 'confidence of lex', 'length', 'period', 'question mark', 'exclamation point', 'ellipses']




df_train = df_train.drop('sentence', axis=1)
df_test = df_ver.drop('sentence', axis=1)
train_data = df_train.to_numpy()
test_data = df_test.to_numpy()
X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]
# y_train = LabelEncoder().fit_transform(y_train)
# y_test = LabelEncoder().fit_transform(y_test)


from sklearn.naive_bayes import CategoricalNB, MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
# cnb = CategoricalNB()
# cnb.fit(X_train, y_train)
#
# # making predictions on the testing set
# prediction = cnb.predict(X_test)
#
# print(metrics.classification_report(y_test, prediction, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division=0.0))
# print(metrics.confusion_matrix(y_test, prediction))
tf_idf = TfidfVectorizer()
X_train_tf = tf_idf.fit_transform(X_train)
X_train_tf = tf_idf.transform(X_train)


X_test_tf = tf_idf.transform(X_test)
naive_bayes_classifier = MultinomialNB(force_alpha=True) #categorical
naive_bayes_classifier.fit(X_train_tf, y_train)
y_pred = naive_bayes_classifier.predict(X_test_tf)
print(metrics.classification_report(y_test, y_pred, labels=None, target_names=None, sample_weight=None, digits=2,
                                      output_dict=False, zero_division=0.0))
print("Confusion matrix:")
print(metrics.confusion_matrix(y_test, y_pred))

