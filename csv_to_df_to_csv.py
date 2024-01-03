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


df = pd.read_csv("MELD_train_split_efr.csv " , sep=',')
print(df)
# df = format(df)
# df.to_csv("MELD_train_split_efr_modif.csv", sep='\t')