import spacy
from sklearn.preprocessing import MultiLabelBinarizer
from lexicon_conversion import data_dict
import numpy as np
from nltk.stem import WordNetLemmatizer
import os
import json
import pandas as pd
from deep_translator import GoogleTranslator
from unlimited_machine_translator.translator import machine_translator_df

possible_emotions = ['disgust', 'fear', 'sadness', 'anger', 'joy', 'neutral', 'surprise', 'contempt']

nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()


def make_training_df(data):
    """
    Extract the sentences and the emotions associated with them, then translate the sentences.
    """
    print("making df")
    sentences = [sentence for group in [episode["utterances"] for episode in data] for sentence in group]
    labels = [label for group in [episode["emotions"] for episode in data] for label in group]
    new_sentences = []
    new_labels = []
    for sentence in sentences:
        if sentence not in new_sentences:
            new_sentences.append(sentence)
            index = sentences.index(sentence)
            emotion_name = labels[index]
            new_labels.append(emotion_name)
    sentences = new_sentences
    labels = new_labels

    df = pd.DataFrame({"label": labels, "sentence": sentences})
    df = df.drop_duplicates().reset_index(drop=True)
    sentences = df['sentence'].tolist()
    labels = df['label'].tolist()
    print("translating...")
    translated = GoogleTranslator('hi', 'en').translate_batch(sentences)
    df = pd.DataFrame({"label": labels, "sentence": translated})
    print("translation complete.")
    return df


def make_test_df(data):
    print("making df")
    sentences = [sentence for group in [episode["utterances"] for episode in data] for sentence in group]
    # labels = [label for group in [episode["emotions"] for episode in data] for label in group]
    new_sentences = []
    # new_labels = []
    for sentence in sentences:
        # if sentence not in new_sentences:
            new_sentences.append(sentence)
            index = sentences.index(sentence)
            # emotion_name = labels[index]
            # new_labels.append(emotion_name)
    sentences = new_sentences
    # labels = new_labels

    df = pd.DataFrame({ "sentence": sentences})
    df["labels"] = 'neutral'
    # df = df.drop_duplicates().reset_index(drop=True)
    sentences = df['sentence'].tolist()
    print(f"len sentences before translation: {len(sentences)}") # check
    print("translating...")
    translated = GoogleTranslator('hi', 'en').translate_batch(sentences)
    print(f"len sentences after translation: {len(translated)}")
    # print(len(labels))
    # print(len(sentences))
    # print(len(translated))
    df = pd.DataFrame({ "sentence": translated})
    df['label'] = 'neutral'
    print("translation complete.")
    return df


def add_features_to_df(df, has_label):
    print("Formatting...")
    labels = df['label'].tolist()
    sentences = df['sentence'].tolist()
    def sentence_into_lemma_list(sentence):
        sentence = nlp(sentence)
        lemmatized_words = []
        for token in sentence:
            word = token.lemma_
            lemmatized_words.append(word)
        return lemmatized_words

    def get_features():
        """
        Extract features from translated sentences.
        """
        punct_list = ['.', '?', '!', '...', '....', '.....']

        confidence_list = []

        positive_list = []
        negative_list = []

        len_values = []

        period_list = []
        question_list = []
        exclamation_list = []
        ellipses_list = []

        prediction_list = []
        justification_list = []

        no_sentences_list = []

        for line in sentences:

            lemmas = sentence_into_lemma_list(line)
            len_values.append(len(lemmas))

            pos = 0
            neg = 0

            period = 0
            question = 0
            exclamation = 0
            ellipses = 0

            anger = 0
            disgust = 0
            fear = 0
            joy = 0
            sadness = 0
            surprise = 0

            justification = []

            no_sentences = 0

            for item in lemmas:

                if item in data_dict: # if item in lexicon

                    if data_dict[item]['anger']:
                        anger += 1
                        justification.append([item, 'anger'])
                    if data_dict[item]['disgust']:
                        disgust += 1
                        justification.append([item, 'disgust'])
                    if data_dict[item]['fear']:
                        fear += 1
                        justification.append([item, 'fear'])
                    if data_dict[item]['joy']:
                        joy += 1
                        justification.append([item, 'joy'])
                    if data_dict[item]['sadness']:
                        sadness += 1
                        justification.append([item, 'sadness'])
                    if data_dict[item]['surprise']:
                        anger += 1
                        justification.append([item, 'surprise'])

                    if data_dict[item]['positive']:
                        pos += 1
                    if data_dict[item]['negative']:
                        neg += 1

                else:
                    if item in punct_list:
                        if lemmas[lemmas.index(item)-1]:
                            no_sentences += 1
                    if item == '.':
                        period += 1
                    if item == '?':
                        question += 1
                    if item == '!':
                        exclamation += 1
                    if item == '...' or item == '....' or item == '.....':
                        ellipses += 1

            if lemmas[-1] not in punct_list:
                no_sentences += 1

            no_sentences_list.append(no_sentences/10)

            total_punctuation = period + question + exclamation + ellipses
            if total_punctuation:
                period_list.append(period / total_punctuation)
                question_list.append(question / total_punctuation)
                exclamation_list.append(exclamation / total_punctuation)
                ellipses_list.append(ellipses / total_punctuation)
            else:
                period_list.append(0.0)
                question_list.append(0.0)
                exclamation_list.append(0.0)
                ellipses_list.append(0.0)

            total_sentiments = pos + neg
            if total_sentiments:
                pos /= total_sentiments
                neg /= total_sentiments
            positive_list.append(pos)
            negative_list.append(neg)

            all_emotion_instances = joy + anger + sadness + fear + disgust + surprise
            max_value_of_any_emotion = max(joy, anger, sadness, fear, disgust, surprise)
            prediction = []
            if max_value_of_any_emotion == joy:
                prediction.append('joy')
            if max_value_of_any_emotion == anger:
                prediction.append('anger')
            if max_value_of_any_emotion == sadness:
                prediction.append('sadness')
            if max_value_of_any_emotion == fear:
                prediction.append('fear')
            if max_value_of_any_emotion == disgust:
                prediction.append('disgust')
            if max_value_of_any_emotion == surprise:
                prediction.append('surprise')
            if all_emotion_instances:
                confidence = max_value_of_any_emotion / all_emotion_instances
            else:
                prediction = ['neutral']
                confidence = 1.0

            prediction_list.append(prediction)
            confidence_list.append(confidence)
            justification_list.append(justification)

        max_len = max(len_values)
        for value in len_values:
            len_values[len_values.index(value)] /= max_len

        return prediction_list, justification_list, confidence_list, positive_list, negative_list, len_values, period_list, question_list, exclamation_list, ellipses_list, no_sentences_list

    predict, just, conf, pos_list, neg_list, lengths, periods, questions, exclamations, ellipses, no_sestences = get_features()

    data_for_df_init = {'sentence': sentences,
                        'number of sentences': no_sestences,
                        'positive': pos_list,
                        'negative': neg_list,
                        'confidence of lex': conf,
                        'length': lengths,
                        'period': periods,
                        'question mark': questions,
                        'exclamation point': exclamations,
                        'ellipses': ellipses,
                        }
    df_init = pd.DataFrame.from_dict(data_for_df_init)

    emotion_series = pd.Series(predict)
    mlb = MultiLabelBinarizer()
    one_hot_emotions = pd.DataFrame(mlb.fit_transform(emotion_series),
                                    columns=mlb.classes_,
                                    index=emotion_series.index)


    for emotion in possible_emotions:
        if emotion not in one_hot_emotions.columns:
            one_hot_emotions[emotion] = 0
    df_init['index'] = np.arange(1, df_init.shape[0] + 1)
    one_hot_emotions['index'] = np.arange(1, one_hot_emotions.shape[0] + 1)


    os.makedirs('final_data', exist_ok=True)
    # df_init.to_csv('data1/final/init.csv')
    # one_hot_emotions.to_csv('data1/final/one_hot.csv')

    df_final = pd.merge(one_hot_emotions, df_init, on='index')
    if has_label:
        df_final['label'] = labels
    else:
        df_final['label'] = 'neutral'
    df_final.to_csv('data1/final/final_df.csv')

    return df_final


with open("MaSaC_train_erc.json", "r") as fp: train_data = json.load(fp)  # MELD_train_efr.json/T3-sample.json
with open("MaSaC_test_erc.json", "r") as fp: test_data = json.load(fp)  # MELD_train_efr.json/T3-sample.json


df_train = make_training_df(train_data)
df_test = make_test_df(test_data)

df_train = add_features_to_df(df_train, True)
df_test = add_features_to_df(df_test, True)

os.makedirs('final_data', exist_ok=True)

df_train.to_csv('final_data/train_data_dtrf.csv', sep=',')
df_test.to_csv('final_data/test_data_dtrf.csv', sep=',')

# Use this instead if preprocessing has already been made
# df_train = pd.read_csv('final_data/train_data_dtrf.csv')
# df_test = pd.read_csv('final_data/test_data_dtrf.csv')

