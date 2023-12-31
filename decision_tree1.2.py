import pandas as pd
from main import df_train
from main import df_ver
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

features = ['anger', 'surprise','disgust', 'fear', 'joy', 'neutral', 'sadness', 'positive', 'negative', 'confidence of lex', 'length', 'period', 'question mark', 'exclamation point', 'ellipses']

features1 = ['anger','surprise','disgust', 'fear', 'joy', 'neutral', 'sadness', 'positive', 'negative', 'confidence of lex', 'length', 'period', 'question mark', 'exclamation point', 'ellipses']

# df_train.head()
# df_ ver.head()

data = df_train.to_numpy()
ver_data = df_ver.to_numpy()

X = df_train[features1]
y = df_train['label']

X_ver = df_ver[features1]
y_ver = df_ver['label']

y = LabelEncoder().fit_transform(y)
y_ver = LabelEncoder().fit_transform(y_ver)

# df_train = df_train.drop('sentence', axis=1)
# df_ver = df_ver.drop('sentence', axis=1)
# train_data = df_train.to_numpy()
# ver_data = df_ver.to_numpy()
# X_train, y_train = train_data[:, :-1], train_data[:, -1]
# X_ver, y_ver = ver_data[:, :-1], ver_data[:, -1]
# y_train = LabelEncoder().fit_transform(y_train)
# y_ver = LabelEncoder().fit_transform(y_ver)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = tree.DecisionTreeClassifier(class_weight='balanced')
steps = [('over', SMOTE()), ('model', model)]
pipeline = Pipeline(steps=steps)

pipeline.fit(X_train, y_train)

prediction = pipeline.predict(X_ver)
print("F1 FOR VER")
print(metrics.classification_report(y_ver, prediction, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division=0.0))
print(metrics.confusion_matrix(y_ver, prediction))

prediction = pipeline.predict(X_test)
print("F1 FOR TEST")
print(metrics.classification_report(y_test, prediction, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division=0.0))
print(metrics.confusion_matrix(y_test, prediction))

# imp_df = pd.DataFrame({
#     "Varname": X_train.columns,
#     "Imp": model.feature_importances_
# })
#
# imp_df.sort_values(by="Imp", ascending=False)
# print(imp_df)