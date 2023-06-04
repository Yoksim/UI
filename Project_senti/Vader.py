# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:35:49 2021

@author: zhendahu
"""
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import nltk
nltk.download('vader_lexicon')
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd 
import numpy as np 
import os 

df = pd.read_csv('C:/Users/zhendahu/Desktop/000workwell - to zhenda/validation with other existing models/sg-transport-1115-clean.csv', header=0, encoding='utf-8')
#df = pd.read_csv('C:/Users/zhendahu/Desktop/Project_senti/0test-ready-1.csv')
#df = pd.read_csv('C:/Users/zhendahu/Desktop/Project_senti/new_raw_movie.csv')


df['Text'] = df['Text'].str.lower()

df = df[['Text', 'Sentiment', 'Sentiment Num', 'Multi']].copy()
df['Sentiment'] = df['Sentiment'].str.lower()

analyzer = SIA()
'''
for s in df['Text']:
    vs = analyzer.polarity_scores(str(s))
    print("{:-<65} {}".format(s, str(vs)))
'''

sentences= df["Text"]
for i in range(len(sentences)):
    scores = analyzer.polarity_scores(str(sentences.iloc[i]))
    #print (scores)

#Printing the sentiment nicely in a table format. 
my_vader_score_compound = [ ] 
my_vader_score_positive = [ ] 
my_vader_score_negative = [ ] 
my_vader_score_neutral = [ ] 

for i in range(len(sentences)):
    my_analyzer = analyzer.polarity_scores(str(sentences.iloc[i]))
    my_vader_score_compound.append(my_analyzer['compound'])
    my_vader_score_positive.append(my_analyzer['pos'])
    my_vader_score_negative.append(my_analyzer['neg']) 
    my_vader_score_neutral.append(my_analyzer['neu']) 


#converting sentiment values to numpy for easier usage
my_vader_score_compound = np.array(my_vader_score_compound)
my_vader_score_positive = np.array(my_vader_score_positive)
my_vader_score_negative = np.array(my_vader_score_negative)
my_vader_score_neutral = np.array(my_vader_score_neutral)

# neg and positive given higher weightage than neutral probabily
df['polarity'] = my_vader_score_compound
df['postive'] = my_vader_score_positive
df['neg'] = my_vader_score_negative
df['neu'] = my_vader_score_neutral

# This option is just to restrict the column width for printing purposes.
#pd.options.display.max_colwidth = 40
# Print the dataframe
#print(df[:10])

# just the normal vader
#3-class

df['vader sentiment'] = 0.0
df.loc[(df['polarity'] > 0) & (df['polarity'] <= 1), 'vader sentiment'] = 1.0
df.loc[(df['polarity'] < 0), 'vader sentiment'] = -1.0

#5-class
'''
df['vader sentiment'] = 0.0
df.loc[(df['polarity'] > 0.5) & (df['polarity'] <= 1), 'vader sentiment'] = 2.0
df.loc[(df['polarity'] > 0) & (df['polarity'] <= 0.5), 'vader sentiment'] = 1.0
df.loc[(df['polarity'] < 0) & (df['polarity'] >= -0.5), 'vader sentiment'] = -1.0
df.loc[(df['polarity'] < -0.5) & (df['polarity'] >= -1), 'vader sentiment'] = -2.0
'''

print(accuracy_score(df['Sentiment Num'], df["vader sentiment"]))
print(precision_score(df['Sentiment Num'], df['vader sentiment'], average='weighted'))
print(recall_score(df['Sentiment Num'], df['vader sentiment'], average='weighted'))
print(f1_score(df['Sentiment Num'], df['vader sentiment'], average='weighted'))