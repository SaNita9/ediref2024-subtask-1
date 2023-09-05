import pandas as pd
import spacy
import json

nlp = spacy.load("en_core_web_sm")

with open("T3-sample.json","r") as fp: data = json.load(fp)

sentences = [sentence for group in [episode["utterances"] for episode in data] for sentence in group]
labels = [label for group in [episode["emotions"] for episode in data] for label in group]
new_sentences =[]
new_labels = []
for sentence in sentences:
    if sentence not in new_sentences:
        new_sentences.append(sentence)
        index = sentences.index(sentence)
        new_labels.append(labels[index])
sentences = new_sentences
labels = new_labels

df = pd.DataFrame({"Label":labels,"Sentence":sentences})
df = df.drop_duplicates().reset_index(drop=True)

lexicon = pd.read_csv("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", sep='\t', header=None, names=["Word","Emotion","Value"])
lexicon2=lexicon.pivot(index="Word",columns="Emotion",values="Value")
lexicon2.reset_index(inplace=True)
lexicon2=lexicon2.drop(columns=["negative","positive","trust"])

dict = lexicon.to_dict('split')
del dict['index']
del dict['columns']
data_list = dict['data']



def lexicon_search():
    no_examples = 0
    got_right = 0
    neutrals = 0
    for sentence in sentences:
        ac_emotions = labels[sentences.index(sentence)]
        print(f"\n{no_examples + 1}")
        print(sentence)
        sentence = nlp(sentence)
        anger = 0
        anticipation = 0
        disgust = 0
        fear = 0
        joy = 0
        sadness = 0
        trust = 0
        lemmatized_tokens = []
        justification = []
        for token in sentence:
            word = token.lemma_
            lemmatized_tokens.append(word)
            if [word, 'anger', 0] in data_list or [word, 'anger', 1] in data_list:
                if [word, 'anger', 1] in data_list:
                    anger += 1
                    justification.append([word, 'anger', 1])
                if [word, 'anticipation', 1] in data_list:
                    anticipation += 1
                    justification.append([word, 'anticipation', 1])
                if [word, 'disgust', 1] in data_list:
                    disgust += 1
                    justification.append([word, 'disgust', 1])
                if [word, 'fear', 1] in data_list:
                    fear += 1
                    justification.append([word, 'fear', 1])
                if [word, 'joy', 1] in data_list:
                    joy += 1
                    justification.append([word, 'joy', 1])
                if [word, 'sadness', 1] in data_list:
                    sadness += 1
                    justification.append([word, 'sadness', 1])
                if [word, 'trust', 1] in data_list:
                    trust += 1
                    justification.append([word, 'trust', 1])
        print(lemmatized_tokens)
        all_emotion_instances = joy + anger + sadness + fear + disgust + trust + anticipation
        max_value_of_any_emotion = max(joy, anger, sadness, fear, disgust, trust, anticipation)
        prediction = []
        if max_value_of_any_emotion == joy:
            prediction.append('joy')
        if max_value_of_any_emotion == anger:
            prediction.append('anger')
        if max_value_of_any_emotion == sadness:
            prediction.append('sadness')
        if max_value_of_any_emotion == fear:
            prediction.append('fear')
        if max_value_of_any_emotion == anticipation:
            prediction.append('anticipation')
        if max_value_of_any_emotion == trust:
            prediction.append('trust')
        if max_value_of_any_emotion == disgust:
            prediction.append('disgust')
        if ac_emotions == 'neutral':
            neutrals += 1

        if all_emotion_instances:
            confidence = max_value_of_any_emotion / all_emotion_instances
        else:
            prediction = 'neutral'
            confidence = 1.0
        print(f"prediction: {prediction}")
        print(f"confidence: {confidence}")
        print(f"based on: {justification}")
        print(f"actual emotion: {ac_emotions}")
        if ac_emotions in prediction:
            got_right += 1
        no_examples +=1

    accuracy = got_right / no_examples * 100
    comparison = neutrals / no_examples * 100
    print(f'\naccuracy: {accuracy}%')
    print(f'accuracy if we just assumed every utterance was neutral: {comparison}%')


lexicon_search()


