from main import df_train
from sklearn.model_selection import train_test_split
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import BaggingClassifier
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from main import df_train
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib

# example of oversampling a multi-class classification dataset
from pandas import read_csv
from imblearn.over_sampling import SMOTE
from collections import Counter
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
df_train.head()

features = ['anger',  'disgust', 'fear', 'joy',  'sadness', 'positive',
            'negative', 'confidence of lex', 'length', 'period', 'question mark', 'exclamation point', 'ellipses']

x0 = df_train[features]
y0 = df_train['label']
X_train, X_test, y_train, y_test = train_test_split(x0, y0, test_size=0.2, random_state=42)
X=X_train
y=y_train.to_numpy()


# load and summarize the dataset
from pandas import read_csv
from collections import Counter
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
# define the dataset location




y = LabelEncoder().fit_transform(y)
# transform the dataset
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
# summarize distribution
counter = Counter(y)
# for k,v in counter.items():
#  per = v / len(y) * 100
#  print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# # plot the distribution
# pyplot.bar(counter.keys(), counter.values())
# pyplot.show()
classifier_rf = RandomForestClassifier(max_depth=9, min_samples_leaf=19, n_estimators=37,
                                        n_jobs=-1, random_state=42, oob_score=True)
classifier_rf.fit(X_train, y_train)

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
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
# #print(rf_best)
print(classifier_rf.oob_score_)
pred =classifier_rf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, pred))
#print("prec:", metrics.precision_score(y_test, pred, ))
#print("rec:", metrics.recall_score(y_test, pred))
#print("f1:", metrics.f1_score(y_test, pred))#####de copiat de la decision tree ca e prost criteriul
# rf.fit(X_train, y_train)
# imp_df = pd.DataFrame({
#     "Varname": X_train.columns,
#     "Imp": rf.feature_importances_
# })
# imp_df.sort_values(by="Imp", ascending=False)
# print(imp_df)