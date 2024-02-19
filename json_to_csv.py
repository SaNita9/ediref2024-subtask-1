"""
Transforms the jsons provided into csv documents used in the training and testing of BERT
"""

import pandas as pd
import numpy as np
import json
import os

def json_to_df(df, has_label):
    """
    Extract the sentences and the emotions associated with them for BERT.
    """
    if has_label:
        emotions = df['emotions'].tolist()
        labels = []
        for list in emotions:
            for emotion in list:
                labels.append(emotion)
    sentences = df['utterances'].tolist()
    text = []
    for list in sentences:
        for sentence in list:
            text.append(sentence)
    if has_label:
        df = pd.DataFrame({"text": text, "label": labels})
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        df = pd.DataFrame({"text": text})
        df["label"] = 'neutral'

    return df

with open('MaSaC_val_erc.json', encoding='utf-8') as inputfile:
    df_val = pd.read_json(inputfile)
with open('MaSaC_train_erc.json', encoding='utf-8') as inputfile:
    df_train = pd.read_json(inputfile)
with open('MaSaC_test_erc.json', encoding='utf-8') as inputfile:
    df_test = pd.read_json(inputfile)

df_val = json_to_df(df_val, True)
df_train = json_to_df(df_train, True)
df_test = json_to_df(df_test, False)

os.makedirs('final_data', exist_ok=True)

df_val.to_csv("final_data/val_data_bert.csv", sep=',', encoding='utf-8', index=False, header=True)
df_train.to_csv("final_data/train_data_bert.csv", sep=',', encoding='utf-8', index=False, header=True)
df_test.to_csv("final_data/test_data_bert.csv", sep=',', encoding='utf-8', index=False, header=True)
