from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from preprocessing_dt_rf import df_train, df_test
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

features = ['anger', 'surprise','disgust', 'fear', 'joy', 'neutral', 'sadness', 'positive', 'negative', 'confidence of lex', 'length', 'period', 'question mark', 'exclamation point', 'ellipses']
train_data = df_train.to_numpy()
ver_data = df_test.to_numpy()

def X_y(df):
    x = df[features]
    y = df['label']
    y = LabelEncoder().fit_transform(y)
    return x, y

X_train, y_train = X_y(df_train)
X_test, y_test = X_y(df_test)

model= tree.DecisionTreeClassifier(max_depth=3, class_weight='balanced')
steps = [('over', SMOTE()), ('model', model)]
dt_pipeline = Pipeline(steps=steps)
dt_pipeline.fit(X_train, y_train)
dt_prediction = dt_pipeline.predict(X_test)

rf = RandomForestClassifier(max_depth=10, min_samples_leaf=19, n_estimators=30,
                            n_jobs=-1, random_state=42, oob_score=True, class_weight='balanced')
rf.fit(X_train, y_train)
rf_prediction = rf.predict(X_test)
