import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from main import df_final
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib

features = ['anger', 'surprise','disgust', 'fear', 'joy', 'neutral', 'sadness', 'positive', 'negative', 'confidence of lex', 'length', 'period', 'question mark', 'exclamation point', 'ellipses']

df_final.head()

# Separate Target Variable and Predictor Variables

X = df_final[features]
y = df_final['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(X_train.shape, X_test.shape)

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

rf.fit(X_train, y_train)
imp_df = pd.DataFrame({
    "Varname": X_train.columns,
    "Imp": rf.feature_importances_
})
imp_df.sort_values(by="Imp", ascending=False)
print(imp_df)