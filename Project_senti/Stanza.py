# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 18:53:11 2021

@author: zhendahu
"""

import stanza
stanza.download('en') 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd 
import statistics

#df = pd.read_csv('C:/Users/zhendahu/Desktop/000workwell - to zhenda/validation with other existing models/sg-transport-1115-clean.csv', header=0, encoding='utf-8')
df = pd.read_csv('C:/Users/zhendahu/Desktop/Project_senti/0test-ready-1.csv')
df = pd.read_csv('C:/Users/zhendahu/Desktop/Project_senti/new_raw_movie.csv')

nlp = stanza.Pipeline(lang='en', processors='tokenize,sentiment')

def stanford_nlp_score(text):
    doc = nlp(text)
    list_of_polarities = []
    for i, sentence in enumerate(doc.sentences):
        list_of_polarities.append(sentence.sentiment)
    return statistics.mean(list_of_polarities)

df["Predicted Polarity"] = df['Text'].apply(lambda text: stanford_nlp_score(text))

'''
df['Predicted Polarity'] = round(df['Predicted Polarity'])
df['Predicted Polarity'] = df['Predicted Polarity'].astype(int)

sentiment_name = 'Sentiment Num'
def replace(value):
    if value == 0:
        return -1
    elif value == 1:
        return 0
    elif value == 2:
        return 1
df['Predicted Polarity'] = df['Predicted Polarity'] .apply(replace)

print(accuracy_score(df[sentiment_name], df['Predicted Polarity']))
print(precision_score(df[sentiment_name], df['Predicted Polarity'], average='weighted'))
print(recall_score(df[sentiment_name], df['Predicted Polarity'], average='weighted'))
print(f1_score(df[sentiment_name], df['Predicted Polarity'], average='weighted'))
'''