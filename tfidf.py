import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import nltk
import re
import string
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
import argparse
import os
import json
import sys
import ast

parser = argparse.ArgumentParser()
parser.add_argument("--lemma", help = "Lemmatization", action='store_true')
parser.add_argument("--stop", help = "Remove stop words", action='store_true')
parser.add_argument("--train", help = "Train CSV", required=True)
parser.add_argument("--test", help = "Test CSV", required=True)
parser.add_argument("--out", help = "Output base name for JSON and CSV (if pred)", required=True)
parser.add_argument("--pred", help = "Perform prediction", action='store_true')
args = parser.parse_args()

if not os.path.exists("data/pred_tfidf"):
    os.makedirs("data/pred_tfidf")

if args.stop:
    stopwords = nltk.corpus.stopwords.words('english')
    # nltk.download('stopwords')

if args.lemma:
    # nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()

def format(df):
    # text = df['utterances'].tolist()
    # text = ast.literal_eval(text)
    # text = [item for list in text for item in list]
    emotions = df['emotions'].tolist()
    labels = []
    for list in emotions:
        list = ast.literal_eval(list)
        for emotion in list:
            labels.append(emotion)
    sentences = df['utterances'].tolist()
    text = []
    for list in sentences:
        list = ast.literal_eval(list)
        for sentence in list:
            text.append(sentence)

    df = pd.DataFrame({"text": text, "label": labels})
    df = df.drop_duplicates().reset_index(drop=True)

    return df

def make_X_y(train):
    train_X_text = train['text'].to_numpy()
    train_y = train['label'].to_numpy()
    train_X = []

    for i in range(0,len(train_X_text)):
        text=train_X_text[i]
        if args.stop or args.lemma:
            text=text.split()
            if args.stop:
                text=[word for word in text if not word in set(stopwords)]
            if args.lemma:
                text=[lemmatizer.lemmatize(word) for word in text]
            text=' '.join(text)
        train_X.append(text)

    return (train_X, train_y)

dataTrain = pd.read_csv(args.train, sep=',')
dataTrain = format(dataTrain)
labels=[str(l) for l in dataTrain.label.unique()]

dataTest = pd.read_csv(args.test, sep=',')
dataTest = format(dataTest)

(train_X, train_y)=make_X_y(dataTrain)
(test_X, test_y)=make_X_y(dataTest)

tf_idf = TfidfVectorizer()
X_train_tf = tf_idf.fit_transform(train_X)
X_train_tf = tf_idf.transform(train_X)

X_test_tf = tf_idf.transform(test_X)

print("TRAIN: n_samples: %d, n_features: %d" % X_train_tf.shape)
print("TEST: n_samples: %d, n_features: %d" % X_test_tf.shape)

naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_tf, train_y)
y_pred = naive_bayes_classifier.predict(X_test_tf)

with open("data/pred_tfidf/{out}.params.json".format(out=args.out),"w") as f:
    json.dump(vars(args),f,indent=4)

if not args.pred:
    print(metrics.classification_report(test_y, y_pred, target_names=labels, sample_weight=None, digits=2,
                                      output_dict=False, zero_division=0.0))

    print("Confusion matrix:")
    print(metrics.confusion_matrix(test_y, y_pred))

    res=metrics.classification_report(test_y, y_pred, target_names=labels, output_dict=True, zero_division=0.0)
    with open("data/pred_tfidf/{out}.json".format(out=args.out),"w") as f:
        json.dump(res,f,indent=4)

else:
    dataTest['pred']=y_pred
    dataTest.to_csv('data/pred_tfidf/{out}.pred'.format(out=args.out),sep='\t', index=False)