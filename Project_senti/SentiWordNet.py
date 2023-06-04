# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:21:03 2021

@author: zhendahu
"""

import pandas as pd
import nltk
nltk.download('sentiwordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import sentiwordnet as swn
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score

#df = pd.read_csv('C:/Users/zhendahu/Desktop/000workwell - to zhenda/validation with other existing models/sg-transport-1115-clean.csv', header=0, encoding='utf-8')
df = pd.read_csv('C:/Users/zhendahu/Desktop/Project_senti/0test-ready-1.csv')
df = pd.read_csv('C:/Users/zhendahu/Desktop/Project_senti/new_raw_movie.csv')

df['Text'] = df['Text'].str.lower()
df=df[['Sentiment Num','Text', 'Multi']]
df['Text'].dtypes
df['Text'] = df['Text'].astype(str)
#print(df.head())


def sentiment_calc(doc):
    try:
        sentences = nltk.sent_tokenize(doc)
        stokens = [nltk.word_tokenize(sent) for sent in sentences]
        taggedlist=[]
        for stoken in stokens:        
             taggedlist.append(nltk.pos_tag(stoken))
        wnl = nltk.WordNetLemmatizer()

        score_list=[]
        for idx,taggedsent in enumerate(taggedlist):
            score_list.append([])
            for idx2,t in enumerate(taggedsent):
                newtag=''
                lemmatized=wnl.lemmatize(t[0])
                if t[1].startswith('NN'):
                    newtag='n'
                elif t[1].startswith('JJ'):
                    newtag='a'
                elif t[1].startswith('V'):
                    newtag='v'
                elif t[1].startswith('R'):
                    newtag='r'
                else:
                    newtag=''       
                if(newtag!=''):    
                    synsets = list(swn.senti_synsets(lemmatized, newtag))
                    #Getting average of all possible sentiments, as you requested        
                    score=0
                    if(len(synsets)>0):
                        for syn in synsets:
                            score+=syn.pos_score()-syn.neg_score()
                        score_list[idx].append(score/len(synsets))
        sentence_sentiment=[]

        def condition(x): 
            return x!=0

        for score_sent in score_list:
            sentence_sentiment.append(sum([word_score for word_score in score_sent])/len(score_sent))
        #print("Sentiment for each sentence for:"+doc)
        #print(sentence_sentiment)

        return sum([sentence for sentence in sentence_sentiment])/len(sentence_sentiment)
    except:
        return None

df['polarity'] = df['Text'].apply(sentiment_calc)

#set the threshold
#3-class

df['new sentiment'] = 0
df.loc[(df['polarity'] > 0) & (df['polarity'] <= 1), 'new sentiment'] = 1.0
df.loc[(df['polarity'] < 0), 'new sentiment'] = -1.0


'''
#5-class
df['new sentiment'] = 0.0
df.loc[(df['polarity'] > 0.1) & (df['polarity'] <= 1), 'new sentiment'] = 2.0
df.loc[(df['polarity'] > 0) & (df['polarity'] <= 0.1), 'new sentiment'] = 1.0
df.loc[(df['polarity'] < 0) & (df['polarity'] >= -0.1), 'new sentiment'] = -1.0
df.loc[(df['polarity'] < -0.1) & (df['polarity'] >= -1), 'new sentiment'] = -2.0
'''

#evaluation
print(accuracy_score(df['Sentiment Num'], df["new sentiment"]))
print(precision_score(df['Sentiment Num'], df['new sentiment'], average='weighted'))
print(recall_score(df['Sentiment Num'], df['new sentiment'], average='weighted'))
print(f1_score(df['Sentiment Num'], df['new sentiment'], average='weighted'))