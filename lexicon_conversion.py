import pandas as pd
# from main import data_list
import json
lexicon = pd.read_csv("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", sep='\t', header=None,
                      names=["Word", "Emotion", "Value"])
lexicon2 = lexicon.pivot(index="Word", columns="Emotion", values="Value")
lexicon2.reset_index(inplace=True)
lexicon2 = lexicon2.drop(columns=["negative", "positive", "trust", "anticipation"])

dictionary = lexicon.to_dict('split')
del dictionary['index']
del dictionary['columns']
data_list = dictionary['data']
# print(data_list)
data_dict = {}

for item in data_list:
    word = item[0]
    emotion = item[1]
    value = item[2]
    if word not in data_dict:
        data_dict[word] = {emotion: value}
    else:
        data_dict[word][emotion] = value

########################################################
# import pandas as pd
# from nltk.stem import WordNetLemmatizer
# # from main import data_list
# import json
# lexicon = pd.read_csv("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt", sep='\t', header=None,
#                       names=["Word", "Emotion", "Value"])
# # lexicon2 = lexicon.pivot(index="Word", columns="Emotion", values="Value")
# # lexicon2.reset_index(inplace=True)
# # lexicon2 = lexicon2.drop(columns=["negative", "positive", "trust", "anticipation"])
#
# dictionary = lexicon.to_dict('split')
# del dictionary['index']
# del dictionary['columns']
# data_list = dictionary['data']
# # print(data_list)
# data_dict_lemmas = {}
# data_dict = {}
# lemmatizer = WordNetLemmatizer()
#
# # for item in data_list:
# #     word = str(item[0])
# #     emotion = item[1]
# #     value = item[2]
# #     lemma = lemmatizer.lemmatize(word)
# #     if lemma not in data_dict_lemmas:
# #         data_dict_lemmas[lemma] = {emotion: value}
# #     else:
# #         data_dict_lemmas[lemma][emotion] = value
# #     if word not in data_dict:
# #         data_dict[word] = {emotion: value}
# #     else:
# #         data_dict[word][emotion] = value
#
# for item in data_list:
#     word = str(item[0])
#     emotion = item[1]
#     value = item[2]
#     lemma = lemmatizer.lemmatize(word)
#     if lemma not in data_dict_lemmas:
#         data_dict_lemmas[lemma] = {emotion: value}
#     else:
#         data_dict_lemmas[lemma][emotion] = value
#     if word not in data_dict:
#         data_dict[word] = {emotion: value}
#     else:
#         data_dict[word][emotion] = value
#
#
# # for i in data_dict:
# #     if i not in data_dict_lemmas:
# #         j = lemmatizer.lemmatize(str(i))
# #         if j in data_dict:
# #             print(j, data_dict[j])
# #         else:
# #             print(j, "is just lemma")
# #         print(i, data_dict[i])
# #         print("")

########################################################