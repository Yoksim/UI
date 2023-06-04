# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:10:06 2021

@author: zhendahu
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from textblob import TextBlob
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score

df = pd.read_csv('C:/Users/zhendahu/Desktop/Project_senti/sg-transport-1115-clean.csv', header=0, encoding='utf-8')
#df = pd.read_csv('C:/Users/zhendahu/Desktop/Project_senti/0test-ready-1.csv')
df = pd.read_csv('C:/Users/zhendahu/Desktop/Project_senti/new_raw_movie.csv')
#df.head()

sentiment_name = 'Multi'
#sentiment_name = 'Sentiment Num'
'''
df.loc[df['Sentiment'].str.contains('neutral'), sentiment_name] = 0.0
df.loc[df['Sentiment'].str.contains('positive'), sentiment_name] = 1.0
df.loc[df['Sentiment'].str.contains('negative'), sentiment_name] = -1.0
'''
df=df[[sentiment_name,'Text']]
df['Text'] = df['Text'].str.lower()
df['Text'].dtypes
df['Text'] = df['Text'].astype(str)



def sentiment_calc(text):
    try:
        return TextBlob(text).sentiment
    except:
        return None

df['polarity'] = df['Text'].apply(sentiment_calc)
df['polarity2'] = df['Text'].apply(lambda x: TextBlob(x).sentiment[0])

df['textblob sentiment'] = 0

df.loc[(df['polarity2'] > 0.2) & (df['polarity2'] <= 1), 'textblob sentiment'] = 1.0
df.loc[(df['polarity2'] < 0), 'textblob sentiment'] = -1.0

'''
df.loc[(df['polarity2'] > 0) & (df['polarity2'] <= 0.5), 'textblob sentiment'] = 1.0
df.loc[(df['polarity2'] > 0.5) & (df['polarity2'] <= 1), 'textblob sentiment'] = 2.0
df.loc[(df['polarity2'] >= -0.5) &(df['polarity2'] < 0), 'textblob sentiment'] = -1.0
df.loc[(df['polarity2'] >= -1) & (df['polarity2'] < -0.5), 'textblob sentiment'] = -2.0
'''

print(accuracy_score(df[sentiment_name], df["textblob sentiment"]))
print(precision_score(df[sentiment_name], df['textblob sentiment'], average='weighted'))
print(recall_score(df[sentiment_name], df['textblob sentiment'], average='weighted'))
print(f1_score(df[sentiment_name], df['textblob sentiment'], average='weighted'))