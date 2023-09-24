import pandas as pd
from main import df_final
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
import matplotlib
features = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'positive', 'negative', 'confidence of lex', 'length', 'period', 'question mark', 'exclamation point', 'ellipses']

df_final.head()

# Separate Target Variable and Predictor Variables
TargetVariable = 'APPROVE_LOAN'
X = df_final[features]
y = df_final['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = tree.DecisionTreeClassifier(max_depth=3, criterion='entropy')

# Printing all the parameters of Decision Trees
print(clf)

# Creating the model on Training Data
DTree = clf.fit(X_train, y_train)
prediction = DTree.predict(X_test)

# Measuring accuracy on Testing Data


print(metrics.classification_report(y_test, prediction, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division=0.0))
print(metrics.confusion_matrix(y_test, prediction))

# Plotting the feature importance for Top 10 most important columns

feature_importances = pd.Series(DTree.feature_importances_, index=features)
feature_importances.nlargest(10).plot(kind='barh')

# Printing some sample values of prediction
TestingDataResults = pd.DataFrame(data=X_test, columns=features)
TestingDataResults['TargetColumn'] = y_test
TestingDataResults['Prediction'] = prediction
TestingDataResults.head()
