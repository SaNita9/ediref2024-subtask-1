import pandas as pd
from main import df_train
from main import df_ver
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
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
y_train = LabelEncoder().fit_transform(y_train)
y_test = LabelEncoder().fit_transform(y_test)

# df_train.head()
#
# X = df_train[features]
# y = df_train['label']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
clf = tree.DecisionTreeClassifier(max_depth=3, criterion='entropy')


print(clf)


DTree = clf.fit(X_train, y_train)
prediction = DTree.predict(X_test)


print(metrics.classification_report(y_test, prediction, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division=0.0))
print(metrics.confusion_matrix(y_test, prediction))


from sklearn.metrics import f1_score

# f2 = f1_score(y_test, prediction, labels=['anger', 'surprise', 'disgust', 'fear', 'joy', 'neutral', 'sadness'], pos_label=1, average=None, sample_weight=None, zero_division='warn')
# print(f2)