import pandas as pd
import spacy
import json
from sklearn.preprocessing import MultiLabelBinarizer
from lexicon_conversion import data_dict

possible_emotions = ['disgust', 'contempt', 'fear', 'sadness', 'anger', 'joy', 'neutral']

nlp = spacy.load("en_core_web_sm")

with open("MELD_train_efr.json", "r") as fp: data = json.load(fp)  # MELD_train_efr.json/T3-sample.json

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

df = pd.DataFrame({"Label": labels, "Sentence": sentences})
df = df.drop_duplicates().reset_index(drop=True)

lexicon = pd.read_csv("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", sep='\t', header=None,
                      names=["Word", "Emotion", "Value"])
lexicon2 = lexicon.pivot(index="Word", columns="Emotion", values="Value")
lexicon2.reset_index(inplace=True)
lexicon2 = lexicon2.drop(columns=["negative", "positive", "trust", "anticipation"])

dictionary = lexicon.to_dict('split')
del dictionary['index']
del dictionary['columns']
data_list = dictionary['data']


def sentence_into_lemma_list(sentence):
    sentence = nlp(sentence)
    lemmatized_tokens = []
    for token in sentence:
        word = token.lemma_
        lemmatized_tokens.append(word)
    return lemmatized_tokens


def get_features():
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

        no_sentences = 0

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

        all_emotion_instances = joy + anger + sadness + fear + disgust
        max_value_of_any_emotion = max(joy, anger, sadness, fear, disgust, )
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
                    # 'prediction from lex': predict,
                    # 'justification from lex': just,
                    }
df_init = pd.DataFrame.from_dict(data_for_df_init)

emotion_series = pd.Series(predict)
mlb = MultiLabelBinarizer()
one_hot_emotions = pd.DataFrame(mlb.fit_transform(emotion_series),
                                columns=mlb.classes_,
                                index=emotion_series.index)
one_hot_emotions.insert(0, 'sentence', sentences)

df_final = pd.merge(one_hot_emotions, df_init, on='sentence')
df_final['label'] = labels

# df_final.to_excel("overview.xlsx")


