import pandas as pd
from sklearn.model_selection import train_test_split
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




def format(df):
    # text = df['utterances'].tolist()
    # text = ast.literal_eval(text)
    # text = [item for list in text for item in list]
    emotions = df['emotions'].tolist()
    labels = []
    for list in emotions:
        # list = ast.literal_eval(list)
        # print(list)
        for emotion in list:
            labels.append(emotion)
    sentences = df['utterances'].tolist()
    text = []
    for list in sentences:
    #     list = ast.literal_eval(list)
        for sentence in list:
            text.append(sentence)
    df = pd.DataFrame({"text": text, "label": labels})
    df = df.drop_duplicates().reset_index(drop=True)

    return df

with open('MELD_train_efr.json', encoding='utf-8') as inputfile:
    df = pd.read_json(inputfile)

df = format(df)

X = df['text']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

df_train = pd.DataFrame({"text": X_train, "label": y_train})
df_test = pd.DataFrame({"text": X_test, "label": y_test})

df_train.to_csv('MELD_train_split_efr.csv', encoding='utf-8', index=False)
df_test.to_csv('MELD_test_split_efr.csv', encoding='utf-8', index=False)
