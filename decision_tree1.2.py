import pandas as pd
from main import df_final
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
import matplotlib
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

df_final = df_final.drop('sentence', axis=1)

data = df_final.to_numpy()

X, y = data[:, :-1], data[:, -1]

y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = tree.DecisionTreeClassifier(class_weight='balanced')
steps = [('over', SMOTE()), ('model', model)]
pipeline = Pipeline(steps=steps)

pipeline.fit(X_train, y_train)

prediction = pipeline.predict(X_test)

print(metrics.classification_report(y_test, prediction, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division=0.0))
print(metrics.confusion_matrix(y_test, prediction))
