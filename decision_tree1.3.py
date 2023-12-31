import pandas as pd
from main import df_train
from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
import matplotlib
from sklearn.datasets import make_classification


from imblearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
# define the dataset location
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/glass.csv'

df_train = df_train.drop('sentence', axis=1)

data = df_train.to_numpy()

X, y = data[:, :-1], data[:, -1]

y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

pca = PCA(n_components=2)
enn = EditedNearestNeighbours()
smote = SMOTE(random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)

model = make_pipeline(pca, enn, smote, knn)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

print(metrics.classification_report(y_test, prediction, labels=None, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division=0.0))
print(metrics.confusion_matrix(y_test, prediction))
