import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from main import df_train
from main import df_ver
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
features = ['anger', 'surprise','disgust', 'fear', 'joy', 'neutral', 'sadness', 'positive', 'negative', 'confidence of lex', 'length', 'period', 'question mark', 'exclamation point', 'ellipses']

features1 = [ 'disgust', 'fear', 'negative',  'length', 'period', 'question mark', 'exclamation point']

df_train.head()
df_ver.head()

train_data = df_train.to_numpy()
test_data = df_ver.to_numpy()

X_train = df_train[features1]
y_train = df_train['label']

X_test = df_ver[features1]
y_test = df_ver['label']
# X_train, y_train = train_data[:, :-1], train_data[:, -1]
# X_test, y_test = test_data[:, :-1], test_data[:, -1]

#X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



rf = RandomForestClassifier(max_depth=10, min_samples_leaf=19, n_estimators=30,
                            n_jobs=-1, random_state=42, oob_score=True, class_weight='balanced')
rf.fit(X_train, y_train)
model = EasyEnsembleClassifier(n_estimators=30,
                            n_jobs=-1, sampling_strategy='all', replacement=True)
rf.fit(X_train, y_train)
# rf = RandomForestClassifier(random_state=42, n_jobs=-1)
# params = {
#     'max_depth': [2,3,5, 7,8, 9, 10, 11, 19,20],
#     'min_samples_leaf': [5,10,18,19, 20, 21, 50],
#     'n_estimators': [25,30,35, 36, 37, 40, 50, 60, 70]
# }
#
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator=rf,
#                            param_grid=params,
#                            cv = 4,
#                            n_jobs=-1, verbose=1, scoring="accuracy")
# grid_search.fit(X_train, y_train)
# rf_best = grid_search.best_estimator_
# print(rf_best)
#print(rf.oob_score_)
imp_df = pd.DataFrame({
    "Varname": X_train.columns,
    "Imp": rf.feature_importances_
})

imp_df.sort_values(by="Imp", ascending=False)
print(imp_df)
prediction = rf.predict(X_test)
print(metrics.classification_report(y_test, prediction, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division=0.0))
print(metrics.confusion_matrix(y_test, prediction))
# rf.fit(X_train, y_train)
