import pandas as pd
from main import df_train
from main import df_ver
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
import matplotlib
# check version number
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from matplotlib import pyplot
from collections import Counter
from sklearn.datasets import make_classification
from matplotlib import pyplot
from numpy import where
features = ['anger', 'surprise', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'positive', 'negative', 'confidence of lex', 'length', 'period', 'question mark', 'exclamation point', 'ellipses']

df_train.head()

# Separate Target Variable and Predictor Variables

X = df_train[features]
y = df_train['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X = df_ver[features]
y = df_ver['label']
# X=X_train
# y=y_train.to_numpy()
#counter = Counter(y)
#print(counter)
oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)
# summarize the new class distribution
#counter = Counter(y)
#print(counter)
clf = tree.DecisionTreeClassifier(max_depth=3, criterion='entropy')
#
# # Printing all the parameters of Decision Trees
# print(clf)

# Creating the model on Training Data
# DTree = clf.fit(X_train, y_train)
# print(y, y_train)
DTree = clf.fit(X_train, y_train)
prediction = DTree.predict(X_test)
#
# # Measuring accuracy on Testing Data
#
#
print(metrics.classification_report(y_test, prediction, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division=0.0))
print(metrics.confusion_matrix(y_test, prediction))
#
# # # Plotting the feature importance for Top 10 most important columns
# #
# # feature_importances = pd.Series(DTree.feature_importances_, index=features)
# # feature_importances.nlargest(10).plot(kind='barh')
# # print(feature_importances)
# # # Printing some sample values of prediction
# # TestingDataResults = pd.DataFrame(data=X_test, columns=features)
# # TestingDataResults['TargetColumn'] = y_test
# # TestingDataResults['Prediction'] = prediction
# # TestingDataResults.head()
# # # surpise-ul scade
# from sklearn.metrics import f1_score
#
# f1 = f1_score(y_test, prediction, labels=['anger', 'surprise', 'disgust', 'fear', 'joy', 'neutral', 'sadness'], pos_label=1, average=None, sample_weight=None, zero_division='warn')
# print(f1)