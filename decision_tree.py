import pandas as pd
from main import df_final
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
import matplotlib
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
features = ['anger', 'surprise', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'positive', 'negative', 'confidence of lex', 'length', 'period', 'question mark', 'exclamation point', 'ellipses']

df_final.head()

# Separate Target Variable and Predictor Variables

X = df_final[features]
y = df_final['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = tree.DecisionTreeClassifier(max_depth=3, criterion='entropy')

# Printing all the parameters of Decision Trees
print(clf)

# Creating the model on Training Data
DTree = clf.fit(X_train, y_train)
prediction = DTree.predict(X_test)


print(metrics.classification_report(y_test, prediction, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division=0.0))
print(metrics.confusion_matrix(y_test, prediction))


from sklearn.metrics import f1_score

f2 = f1_score(y_test, prediction, labels=['anger', 'surprise', 'disgust', 'fear', 'joy', 'neutral', 'sadness'], pos_label=1, average=None, sample_weight=None, zero_division='warn')
print(f2)