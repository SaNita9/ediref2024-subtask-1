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

with open("MELD_train_efr.json", "r") as fp: train_data = json.load(fp)  # MELD_train_efr.json/T3-sample.json
with open("MELD_val_efr.json", "r") as fp: test_data = json.load(fp)  # MELD_train_efr.json/T3-sample.json


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

def format(data):
    sentences = [sentence for group in [episode["utterances"] for episode in data] for sentence in group]
    labels = [label for group in [episode["emotions"] for episode in data] for label in group]
    new_sentences = []
    new_labels = []
    for sentence in sentences:
        if sentence not in new_sentences:
            new_sentences.append(sentence)
            index = sentences.index(sentence)
            new_labels.append(labels[index])
    sentences = new_sentences
    labels = new_labels

    df = pd.DataFrame({"text": sentences, "label": labels})
    df = df.drop_duplicates().reset_index(drop=True)

    return df

df_train = format(train_data)
labels=[str(l) for l in df_train.label.unique()]
print(labels)
#
df_test = format(test_data)

(train_X, train_y)=make_X_y(df_train)
(test_X, test_y)=make_X_y(df_test)

tf_idf = TfidfVectorizer()
X_train_tf = tf_idf.fit_transform(train_X)
X_train_tf = tf_idf.transform(train_X)
print(tf_idf.vocabulary_)
X_test_tf = tf_idf.transform(test_X)


# print("TRAIN: n_samples: %d, n_features: %d" % X_train_tf.shape)
# print("TEST: n_samples: %d, n_features: %d" % X_test_tf.shape)
print(X_train_tf)

naive_bayes_classifier = MultinomialNB(force_alpha=True) #categorical
naive_bayes_classifier.fit(X_train_tf, train_y)
y_pred = naive_bayes_classifier.predict(X_test_tf)

with open("data/pred_tfidf/{out}.params.json".format(out=args.out),"w") as f:
    json.dump(vars(args),f,indent=4)

if not args.pred:
    print(metrics.classification_report(test_y, y_pred, labels=None, target_names=None, sample_weight=None, digits=2,
                                      output_dict=False, zero_division=0.0))
    print("Confusion matrix:")
    print(metrics.confusion_matrix(test_y, y_pred))

    res=metrics.classification_report(test_y, y_pred, target_names=labels, output_dict=True, zero_division=0.0)
    with open("data/pred_tfidf/{out}.json".format(out=args.out),"w") as f:
        json.dump(res,f,indent=4)

else:
    df_test['pred']=y_pred
    df_test.to_csv('data/pred_tfidf/{out}.pred'.format(out=args.out),sep='\t', index=False)