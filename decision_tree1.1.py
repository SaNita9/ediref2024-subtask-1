import pandas as pd
from main import df_train
from main import df_ver
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
import matplotlib
from sklearn.datasets import make_classification
from matplotlib import pyplot
from collections import Counter
from matplotlib import pyplot
from numpy import where
from sklearn.preprocessing import LabelEncoder
from numpy import mean
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


#df_train = df_train.drop('sentence', axis=1)

data = df_train.to_numpy()

df_train = df_train.drop('sentence', axis=1)
df_test = df_ver.drop('sentence', axis=1)
train_data = df_train.to_numpy()
test_data = df_test.to_numpy()
X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]
y_train = LabelEncoder().fit_transform(y_train)
y_test = LabelEncoder().fit_transform(y_test)

# X, y = data[:, :-1], data[:, -1]
# # label encode the target variable
# y = LabelEncoder().fit_transform(y)
# # define pipeline
model = tree.DecisionTreeClassifier(class_weight='balanced')
steps = [('over', SMOTE()), ('model', model)]
pipeline = Pipeline(steps=steps)


pipeline.fit(X_train, y_train)
prediction = pipeline.predict(X_test)

print(metrics.classification_report(y_test, prediction, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division=0.0))
print(metrics.confusion_matrix(y_test, prediction))

