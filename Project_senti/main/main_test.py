# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 03:31:56 2021

@author: zhendahu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 19:40:34 2021

@author: zhendahu
"""

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sentiment_explorerVersion7 import *
import re

def pdColumn_to_list_converter(df):
    df_list = df.values.tolist() #produces list of lists
    proper_list = [item for sublist in df_list for item in sublist] #a single list
    return proper_list

#df = pd.read_csv('C:/Users/zhendahu/Desktop/000workwell - to zhenda/main/sg-transport-1115-clean.csv')  #import reviews
df = pd.read_csv('C:/Users/zhendahu/Desktop/Project_senti/0test-ready-1.csv')
#df = pd.concat([data_trainset, data_testset], axis=0, ignore_index = True)
df['Text-original'] = df['Text']
df["Text"] = df['Text'].apply(lambda text: text.lower())
df.head()

df_replacement_words = pd.read_csv('C:/Users/zhendahu/Desktop/000workwell - to zhenda/main/ngram_20170222_ZX2021_replace_words.csv')   ##import replace words ###ngram_20170222_ZX2021_replace_words
df_replacement_words["original_phases"] = df_replacement_words["original_phases"].apply(lambda text: text.lower())
df_replacement_words["target_produced_words1"] = df_replacement_words["target_produced_words1"].apply(lambda text: text.lower())
df_replacement_words["target_produced_words2"] = df_replacement_words["target_produced_words2"].apply(lambda text: text.lower())
df_replacement_words.head()

#Replace Lexicon
replace_words_dic = df_replacement_words.set_index('original_phases').to_dict()['target_produced_words1'] ## create a dic for replacing words


#Removing Punctuations From Data
def newtext(text):
    text = text.strip()
    text = text.replace('/\s\s+/g', ' ') # replace multiple spaces with a single space
    text = text.replace(":)","happy")
    text = text.replace(":(","sad")
    text = re.sub ('\s+', ' ', text)
    text = re.sub('@[^\s]+','',text)  # delete the username
    text = re.sub('&[^\s]+','',text)
    text = re.sub('#[^\s]+','',text)
    text = re.sub('".*?"', '', text)  # delete anything in quotation marks
    text = re.sub('http[s]?://\S+', '', text) # delete urls
    for k, v in replace_words_dic.items():
        if k in text:
            text = re.sub(k,v,text)
    text = text.replace("comfort delgro","comfortdelgro")            
    text = text.replace("as well as","and")
    text = text.replace("as well","also")
    text = text.replace("would like","shall")
    text = text.replace("should have","slightly negative")
    text = text.replace("could have","slightly negative")
    text = text.replace("would have","slightly negative")
    text = text.replace("would be","slightly negative")
    text = text.replace("could be","slightly negative")
    text = text.replace("should be","slightly negative")
    
    text = text.replace("n’t","not")
    text = text.replace("n't","not")
    text = text.replace("don","not")
    text = text.replace("dun","not")
    text = text.replace("’s","is")
    text = text.replace("'s","is")
    text = text.replace("’ve","have")
    text = text.replace("'ve","have")
    text = text.replace("’d","had")
    text = text.replace("'d","had")
    text = text.replace("’ll","will")
    text = text.replace("'ll","will")
    text = text.replace("’re","are")
    text = text.replace("'re","are")
    text = text.replace("’m","am")
    text = text.replace("'m","am")
      
    text = text.replace("should improve","slightly negative")
    text = text.replace("would improve","slightly negative")
    text = text.replace("could improve","slightly negative")
    text = text.replace("would enhance","slightly negative")
    text = text.replace("could enhance","slightly negative")
    text = text.replace("would enhance","slightly negative")
    text = text.replace("to be honest","I will say")
    text = text.replace("it's like","alike")
    text = text.replace("middle finger","slightly negative")
    text = text.replace("snapped","slightly negative")
    text = text.replace("constantly accelerate","slightly negative") 
    text = text.replace("accelerate constantly","slightly negative")
    text = text.replace("accelerate and decelerate constantly","slightly negative") 
    text = text.replace("stopage","negative")
    text = text.replace("constantly accelerate and decelerate","slightly negative") 
    text = text.replace("extension service","good")
    text = text.replace("take advantage","slightly negative")
    text = text.replace("took advantage","slightly negative")
    text = text.replace("taking advantage","slightly negative")
    text = text.replace("takes advantage","slightly negative") 
    #text = text.replace("please help","slightly negative")
    #text = text.replace("please","do")
    text = text.replace("a great deal of","a lot of") 

    text = re.sub("\S*@\S*\s?",'',text)   # delete email address
    text = text.replace('\n', ' ').replace('\r', '')  # Clean up all "\n"
    
    text = re.sub(r"""
               [)(@#&$]+  # Accept one or more copies of punctuation
               \ *           # plus zero or more copies of a space,
               """,
               "",          # and replace it with no space   [,.;@#?!&$]+ 
               text, flags=re.VERBOSE)
    
    text = text.replace('.', ' .') #specially added to maintain fullstop
    text = text.replace('?', ' ?') #specially added to maintain question mark
    text = text.replace('!', ' !') #specially added to maintain exclamation mark
    text = text.replace(',', ' ,') #specially added to maintain exclamation mark
    text = text.replace(';', ' ;') #specially added to maintain exclamation mark
    text = text.replace(':', ' :') #specially added to maintain exclamation mark
    
    text= re.sub(' +', ' ', text)
    #text= re.sub(':', '', text)
    text= re.sub("[:']", '', text)
    #text = re.sub ('\s+', '', text)
    return text.lower()


def newtext_fullstop(text):
    text = text.strip()
    text = text.replace('/\s\s+/g', ' ') # replace multiple spaces with a single space
    text = text.replace(":)","happy")
    text = text.replace(":(","sad")
    text = re.sub ('\s+', ' ', text)
    text = re.sub('@[^\s]+','',text)  # delete the username
    text = re.sub('&[^\s]+','',text)
    text = re.sub('#[^\s]+','',text)
    text = re.sub('".*?"', '', text)  # delete anything in quotation marks
    text = re.sub('http[s]?://\S+', '', text) # delete urls
    for k, v in replace_words_dic.items():
        if k in text:
            text = re.sub(k,v,text)    
    text = text.replace("comfort delgro","comfortdelgro")       
    text = text.replace("as well as","and")
    text = text.replace("as well","also")
    text = text.replace("would like","shall")
    text = text.replace("should have","slightly negative")
    text = text.replace("could have","slightly negative")
    text = text.replace("would have","slightly negative")
    text = text.replace("would be","slightly negative")
    text = text.replace("could be","slightly negative")
    text = text.replace("should be","slightly negative")

    text = text.replace("n’t","not")
    text = text.replace("n't","not")
    text = text.replace("don","not")
    text = text.replace("dun","not")
    text = text.replace("’s","is")
    text = text.replace("'s","is")
    text = text.replace("’ve","have")
    text = text.replace("'ve","have")
    text = text.replace("’d","had")
    text = text.replace("'d","had")
    text = text.replace("’ll","will")
    text = text.replace("'ll","will")
    text = text.replace("’re","are")
    text = text.replace("'re","are")
    text = text.replace("’m","am")
    text = text.replace("'m","am")
      
    text = text.replace("should improve","slightly negative")
    text = text.replace("would improve","slightly negative")
    text = text.replace("could improve","slightly negative")
    text = text.replace("would enhance","slightly negative")
    text = text.replace("could enhance","slightly negative")
    text = text.replace("would enhance","slightly negative")
    text = text.replace("to be honest","I will say")
    text = text.replace("it's like","alike")
    text = text.replace("middle finger","slightly negative")
    text = text.replace("snapped","slightly negative")
    text = text.replace("constantly accelerate","slightly negative") 
    text = text.replace("accelerate constantly","slightly negative")
    text = text.replace("accelerate and decelerate constantly","slightly negative") 
    text = text.replace("stopage","negative")
    text = text.replace("constantly accelerate and decelerate","slightly negative") 
    text = text.replace("extension service","good")
    text = text.replace("take advantage","slightly negative")
    text = text.replace("took advantage","slightly negative")
    text = text.replace("taking advantage","slightly negative")
    text = text.replace("takes advantage","slightly negative")
    #text = text.replace("please help","slightly negative")
    #text = text.replace("please","do") 
    
    text = re.sub("\S*@\S*\s?",'',text)   # delete email address
    text = text.replace('\n', ' ').replace('\r', '')  # Clean up all "\n"
    
    text = re.sub(r"""
               [)(@#&$]+  # Accept one or more copies of punctuation
               \ *           # plus zero or more copies of a space,
               """,
               "",          # and replace it with no space
               text, flags=re.VERBOSE)
    
    text = text.replace('.', ' .') #specially added to maintain fullstop
    text = text.replace('?', ' ?') #specially added to maintain question mark
    text = text.replace('!', ' !') #specially added to maintain exclamation mark
    text = text.replace(',', ' ,') #specially added to maintain exclamation mark
    text = text.replace(';', ' ;') #specially added to maintain exclamation mark
    text = text.replace(':', ' :') #specially added to maintain exclamation mark
    
    text= re.sub(' +', ' ', text)
    #text= re.sub(':', '', text)
    text= re.sub("[:']", '', text)
    #text = re.sub ('\s+', '', text)
    return text.lower()

df["Text_with_fullstop"] = df['Text'].apply(lambda text: newtext_fullstop(text))
df["Text"] = df['Text'].apply(lambda text: newtext(text))
df.drop(df.columns[0], axis=1)


acc_dict = {}
f1_dict = {}
pre_dict = {}
rec_dict = {}
Sentiment_Num = "Sentiment Num"
#Step 1: Using Prof Wang's Standard English Dictionary Only
df["Polarities Found"] = df['Text'].apply(lambda text: findPolarity(' '.join(text.split())))
df["Polarity Count"] = df['Polarities Found'].apply(lambda scores: countPolarity(scores))
acc_dict['Prof Wang Standard English Only'] = accuracy_score(df[Sentiment_Num], df["Polarity Count"])
pre_dict['Prof Wang Standard English Only'] = precision_score(df[Sentiment_Num], df["Polarity Count"], average='weighted')
rec_dict['Prof Wang Standard English Only'] = recall_score(df[Sentiment_Num], df["Polarity Count"], average='weighted')
f1_dict['Prof Wang Standard English Only'] = f1_score(df[Sentiment_Num], df["Polarity Count"], average='weighted')


df_confusion = pd.crosstab(df['Polarity Count'],df["Sentiment Num"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
#df_confusion

df = df.drop(df.columns[-2:], axis=1)


#Step 1: Using Prof Wang's Standard English Dictionary Only + Negation
df_tem = df
df_tem["Polarities Found"] = df_tem['Text'].apply(lambda text: findPolarity(' '.join(text.split())))
df_tem["Polarity Count"] = df_tem['Polarities Found'].apply(lambda scores: countPolarity(scores))
acc_dict['Prof Wang Standard English Only + Negation'] = accuracy_score(df_tem[Sentiment_Num], df_tem["Polarity Count"])
pre_dict['Prof Wang Standard English Only + Negation'] = precision_score(df_tem[Sentiment_Num], df_tem["Polarity Count"], average='weighted')
rec_dict['Prof Wang Standard English Only + Negation'] = recall_score(df_tem[Sentiment_Num], df_tem["Polarity Count"], average='weighted')
f1_dict['Prof Wang Standard English Only + Negation'] = f1_score(df_tem[Sentiment_Num], df_tem["Polarity Count"], average='weighted')





#Step 10: Combined Standard EL + Combined Singlish + Negation + Transport Domain  + Too Handling + Sarcasm + Adversative + Emoji + Multi
adversative_df["Polarities Found"] = adversative_df['Text'].apply(lambda text: findPolarity6(' '.join(text.split())))
adversative_df['Polarity Count-multi'] = adversative_df.apply(lambda scores: multi_value(scores['Polarities Found'],scores['Text'], scores['Polarity Count-after Adversative'], 5), axis=1)
adversative_df["Polarity Count-multi"] = adversative_df.apply(lambda a: qn_mark(a['Text-original'],a['Polarity Count-multi']),axis=1)
adversative_df['Polarity Count-multi'].unique()

#5-class

def new_multi(value):
    if 0<value<=0.5:
        valuenew=1
    elif value==0:
        valuenew=0
    elif 1>=value>0.5:
        valuenew=2
    elif -0.5<=value<0:
        valuenew=-1
    elif -1<=value<-0.5:
        valuenew=-2
    return valuenew

#3-class
'''
def new_multi(value):
    if 0<value<=0.5:
        valuenew=1
    elif value==0:
        valuenew=0
    elif 1>=value>0.5:
        valuenew=1
    elif -0.5<=value<0:
        valuenew=-1
    elif -1<=value<-0.5:
        valuenew=-1
    return valuenew
'''
adversative_df['Polarity Count-multi-1']=adversative_df['Polarity Count-multi'].apply(new_multi)

acc_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain  + Too Handling + Like Handling + Sarcasm + Adversative + Emoji + Multi'] = accuracy_score(adversative_df["Multi"], adversative_df["Polarity Count-multi-1"])
pre_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain  + Too Handling + Like Handling + Sarcasm + Adversative + Emoji + Multi'] = precision_score(adversative_df["Multi"], adversative_df["Polarity Count-multi-1"], average='weighted')
rec_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain  + Too Handling + Like Handling + Sarcasm + Adversative + Emoji + Multi'] = recall_score(adversative_df["Multi"], adversative_df["Polarity Count-multi-1"], average='weighted')
f1_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain  + Too Handling + Like Handling + Sarcasm + Adversative + Emoji + Multi'] = f1_score(adversative_df["Multi"], adversative_df["Polarity Count-multi-1"], average='weighted')

acc_dict