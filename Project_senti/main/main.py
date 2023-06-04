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
#df = pd.read_csv('C:/Users/zhendahu/Desktop/Project_senti/0test-ready-1.csv')
df = pd.read_csv(r'C:\Users\huzhe\Desktop\MiMuSA\Project_senti\new_raw_movie.csv')
df = pd.read_csv(r'C:\Users\huzhe\Desktop\MiMuSA\Data\data_part.csv')
#df = pd.read_excel('C:/Users/zhendahu/Desktop/Data/200Han.xlsx')
#df = pd.concat([data_trainset, data_testset], axis=0, ignore_index = True)
#df = pd.read_csv('C:/Users/zhendahu/Desktop/000workwell - to zhenda/main/sg-transport-1115-clean.csv')  #import reviews



df['Text-original'] = df['Text']
df["Text"] = df['Text'].apply(lambda text: text.lower())
df.head()

df_replacement_words = pd.read_csv('C:/Users/huzhe/Desktop/MiMuSA/000workwell - to zhenda/main/ngram_20170222_ZX2021_replace_words.csv')   ##import replace words ###ngram_20170222_ZX2021_replace_words
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
    
    #text = text.replace("n’t","not")
    #text = text.replace("n't","not")
    #text = text.replace("don","not")
    #text = text.replace("dun","not")
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
    #Add by Zhenda
    text = text.replace("a great deal of","a lot of") 
    text = text.replace("all but","almost")
    text = text.replace("right now","now")
    text = text.replace("look like","look")
    text = text.replace("looks like","look")
    text = text.replace("feels like","feels")
    text = text.replace("great whale","whale")
    text = text.replace("kind of","kinda")
    text = text.replace("plain old popcorn fun","boring")
    text = text.replace("nothing fresh","nothing")
    text = text.replace("blue crush","crush")
    text = text.replace("a disquiet world","a world")
    text = text.replace("there is no denying","in fact")
    text = text.replace("bottomless pit","bottomless-pit")
    text = text.replace("fails to","cannot")
    text = text.replace("narrative clichã©s","narrative-clichã©s")
    text = text.replace("take a nap","take-a-nap")
    text = text.replace("less like","less")
    text = text.replace("in need of polishing","in-need-of-polishing")
    text = text.replace("cartoon monster","cartoon-monster")
    text = text.replace("rough waters","rough-waters")
    text = text.replace("new guy","new-guy")
    text = text.replace("high school","school")
    text = text.replace("saving grace","saving-grace")
    text = text.replace("want to see again","want-to-see-again")
    text = text.replace("love story","love-story")
    text = text.replace("intrepid hero","intrepid-hero")
    text = text.replace("like that","like-that")
    text = text.replace("treasure island","treasure-island")
    text = text.replace("canned tuna","canned-tuna")
    text = text.replace("high school film","high-school-film")
    text = text.replace("take off","take-off")
    text = text.replace("deja vu","deja-vu")
    text = text.replace("explored very deeply","explored-very-deeply")
    text = text.replace("pretty clever","clever")
    text = text.replace("do not waste your money","do-not-waste-your-money")
    

     

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


'''
#Step2 : Using Prof Wang's Standard el + Prof Wang Singlish Dict
df["Polarities Found"] = df['Text'].apply(lambda text: findPolarity1(' '.join(text.split())))
df["Polarity Count"] = df['Polarities Found'].apply(lambda scores: countPolarity1(scores))
acc_dict['Prof Wang Standard English + Prof Wang Singlish'] = accuracy_score(df["Sentiment Num"], df["Polarity Count"])
f1_dict['Prof Wang Standard English + Prof Wang Singlish'] = f1_score(df["Sentiment Num"], df["Polarity Count"], average='weighted')

df_confusion = pd.crosstab(df['Polarity Count'],df["Sentiment Num"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
#df_confusion

df = df.drop(df.columns[-2:], axis=1)
'''


#Step2 : Using Prof Wang's Standard el + Prof Wang Singlish Dict + Negation
df_tem = df
df_tem["Polarities Found"] = df_tem['Text'].apply(lambda text: findPolarity1(' '.join(text.split())))
df_tem["Polarity Count"] = df_tem['Polarities Found'].apply(lambda scores: countPolarity1(scores))
acc_dict['Prof Wang Standard English + Prof Wang Singlish + Negation'] = accuracy_score(df_tem[Sentiment_Num], df_tem["Polarity Count"])
pre_dict['Prof Wang Standard English + Prof Wang Singlish + Negation'] = precision_score(df_tem[Sentiment_Num], df_tem["Polarity Count"], average='weighted')
rec_dict['Prof Wang Standard English + Prof Wang Singlish + Negation'] = recall_score(df_tem[Sentiment_Num], df_tem["Polarity Count"], average='weighted')
f1_dict['Prof Wang Standard English + Prof Wang Singlish + Negation'] = f1_score(df_tem[Sentiment_Num], df_tem["Polarity Count"], average='weighted')


df_confusion = pd.crosstab(df['Polarity Count'],df["Sentiment Num"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
#df_confusion

df = df.drop(df.columns[-2:], axis=1)


'''
#Step 3: Combined Standard EL + Combined Singlish
df["Polarities Found"] = df['Text'].apply(lambda text: findPolarity2(' '.join(text.split())))
df["Polarity Count"] = df['Polarities Found'].apply(lambda scores: countPolarity2(scores))
acc_dict['Combined Standard English + Combined Singlish'] = accuracy_score(df["Sentiment Num"], df["Polarity Count"])
f1_dict['Combined Standard English + Combined Singlish'] = f1_score(df["Sentiment Num"], df["Polarity Count"], average='weighted')

df_confusion = pd.crosstab(df['Polarity Count'],df["Sentiment Num"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
#df_confusion

df = df.drop(df.columns[-2:], axis=1)
'''

#Step 3: Combined Standard EL + Combined Singlish + Negation
df["Polarities Found"] = df['Text'].apply(lambda text: findPolarity2_neg(' '.join(text.split())))
df["Polarity Count"] = df['Polarities Found'].apply(lambda scores: countPolarity2_neg(scores, 7))
acc_dict['Combined Standard English + Combined Singlish + Negation'] = accuracy_score(df[Sentiment_Num], df["Polarity Count"])
pre_dict['Combined Standard English + Combined Singlish + Negation'] = precision_score(df[Sentiment_Num], df["Polarity Count"], average='weighted')
rec_dict['Combined Standard English + Combined Singlish + Negation'] = recall_score(df[Sentiment_Num], df["Polarity Count"], average='weighted')
f1_dict['Combined Standard English + Combined Singlish + Negation'] = f1_score(df[Sentiment_Num], df["Polarity Count"], average='weighted')

df_confusion = pd.crosstab(df['Polarity Count'],df[Sentiment_Num] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
#df_confusion

df = df.drop(df.columns[-2:], axis=1)


'''
#Step 4: Combined Standard EL + Combined Singlish + Transport Domain
df["Polarities Found"] = df['Text'].apply(lambda text: findPolarity3(' '.join(text.split())))
df["Polarity Count"] = df['Polarities Found'].apply(lambda scores: countPolarity3(scores))
acc_dict['Combined Standard English + Combined Singlish + Transport Domain'] = accuracy_score(df["Sentiment Num"], df["Polarity Count"])
f1_dict['Combined Standard English + Combined Singlish + Transport Domain'] = f1_score(df["Sentiment Num"], df["Polarity Count"], average='weighted')

df_confusion = pd.crosstab(df['Polarity Count'],df["Sentiment Num"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
#df_confusion

df = df.drop(df.columns[-2:], axis=1)
'''

#Step 5: Combined Standard EL + Combined Singlish + Negation + Transport Domain
df["Polarities Found"] = df['Text'].apply(lambda text: findPolarity4(' '.join(text.split())))
df["Polarity Count"] = df['Polarities Found'].apply(lambda scores: countPolarity4(scores, 7))
acc_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain '] = accuracy_score(df[Sentiment_Num], df["Polarity Count"])
pre_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain'] = precision_score(df[Sentiment_Num], df["Polarity Count"], average='weighted')
rec_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain'] = recall_score(df[Sentiment_Num], df["Polarity Count"], average='weighted')
f1_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain'] = f1_score(df[Sentiment_Num], df["Polarity Count"], average='weighted')

df_confusion = pd.crosstab(df['Polarity Count'],df[Sentiment_Num] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
#df_confusion

df = df.drop(df.columns[-2:], axis=1)


#Step 6: Combined Standard EL + Combined Singlish + Negation + Transport Domain + Too Handling
df["Polarities Found"] = df['Text'].apply(lambda text: findPolarity4_too(' '.join(text.split())))
df["Polarity Count"] = df['Polarities Found'].apply(lambda scores: countPolarity4(scores, 7))
acc_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain + Too Handling'] = accuracy_score(df[Sentiment_Num], df["Polarity Count"])
pre_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain + Too Handling'] = precision_score(df[Sentiment_Num], df["Polarity Count"], average='weighted')
rec_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain + Too Handling'] = recall_score(df[Sentiment_Num], df["Polarity Count"], average='weighted')
f1_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain + Too Handling'] = f1_score(df[Sentiment_Num], df["Polarity Count"], average='weighted')

df_confusion = pd.crosstab(df['Polarity Count'],df["Sentiment Num"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
#df_confusion

df = df.drop(df.columns[-2:], axis=1)


#Step 6.1: Combined Standard EL + Combined Singlish + Negation + Transport Domain  + Too Handling + Like Handling
df["Polarities Found"] = df['Text'].apply(lambda text: findPolarity4_too_like(' '.join(text.split())))
df["Polarity Count"] = df['Polarities Found'].apply(lambda scores: countPolarity4(scores, 7))
acc_dict['Combined Standard English + Combined Singlish  + Negation + Transport Domain + Too Handling + Like Handling'] = accuracy_score(df[Sentiment_Num], df["Polarity Count"])
pre_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain + Too Handling + Like Handling'] = precision_score(df[Sentiment_Num], df["Polarity Count"], average='weighted')
rec_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain + Too Handling + Like Handling'] = recall_score(df[Sentiment_Num], df["Polarity Count"], average='weighted')
f1_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain + Too Handling + Like Handling'] = f1_score(df[Sentiment_Num], df["Polarity Count"], average='weighted')

df_confusion = pd.crosstab(df['Polarity Count'],df["Sentiment Num"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
#df_confusion


#Step 6.2: Combined Standard EL + Combined Singlish + Negation + Transport Domain  + Too Handling + Like Handling + Question mark Handling
def qn_mark(original_text, polarity):
    fivewoneh=['what','why','who','where','when','how', 'What','Why','Who','Where','When','How']
    if '?' in original_text:
        original_text = original_text.strip()
        if original_text.split(" ")[0] not in fivewoneh:
            polarity=-1
    else:
        polarity=polarity
    return polarity

df["Polarity Count"] = df.apply(lambda a: qn_mark(a['Text-original'],a['Polarity Count']),axis=1)
acc_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain + Too Handling + Like Handling + Qn Mark Handling'] = accuracy_score(df[Sentiment_Num], df["Polarity Count"])
pre_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain + Too Handling + Like Handling + Qn Mark Handling'] = precision_score(df[Sentiment_Num], df["Polarity Count"], average='weighted')
rec_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain + Too Handling + Like Handling + Qn Mark Handling'] = recall_score(df[Sentiment_Num], df["Polarity Count"], average='weighted')
f1_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain + Too Handling + Like Handling + Qn Mark Handling'] = f1_score(df[Sentiment_Num], df["Polarity Count"], average='weighted')

df_confusion = pd.crosstab(df['Polarity Count'],df[Sentiment_Num] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
#df_confusion



#hzd_Step 7: Combined Standard EL + Combined Singlish + Negation + Transport Domain  + Too Handling + Adversative
df["Polarities Found"] = df['Text'].apply(lambda text: findPolarity5(' '.join(text.split())))
df["Adversative Polarity"] = df['Polarities Found'].apply(lambda scores: countPolarity5(scores, 7))
def adversative_present(polarity_list):
    if (8 in polarity_list) or (-8 in polarity_list):
        return 1
    else:
        return 0

# label presence of adversative
df['Adversative Present?']=df['Polarities Found'].apply(lambda pl:adversative_present(pl))


def update_p_after_adversative(present,polaritys, polaritya):
    if present==1:
        return polaritya
    elif present==0:
        return polaritya

df['Polarity Count-after Adversative'] = df.apply(lambda x: update_p_after_adversative(x['Adversative Present?'],x['Polarity Count'],x['Adversative Polarity']),axis=1)

df["Polarity Count-after Adversative"] = df.apply(lambda a: qn_mark(a['Text-original'],a['Polarity Count-after Adversative']),axis=1)
df['Polarity Count-after Adversative'].unique()
adversative_df = df

acc_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain + Too Handling + Like Handling + Adversative'] = accuracy_score(adversative_df[Sentiment_Num], adversative_df["Polarity Count-after Adversative"])
pre_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain + Too Handling + Like Handling + Adversative'] = precision_score(df[Sentiment_Num], df["Polarity Count-after Adversative"], average='weighted')
rec_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain + Too Handling + Like Handling + Adversative'] = recall_score(df[Sentiment_Num], df["Polarity Count-after Adversative"], average='weighted')
f1_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain + Too Handling + Like Handling + Adversative'] = f1_score(df[Sentiment_Num], df["Polarity Count-after Adversative"], average='weighted')

df_confusion = pd.crosstab(df['Polarity Count-after Adversative'],df["Sentiment Num"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
#df_confusion


#hzd_Step 8: Combined Standard EL + Combined Singlish + Negation + Transport Domain  + Too Handling + Adversative + Sarcasm
df["Sarcasm?"] = df['Polarities Found'].apply(lambda row: recognise_sarcasm(row))
df.loc[((df['Adversative Present?'] == 0) & (df['Sarcasm?'] != 0)), 'Polarity Count-after Adversative'] = -1
acc_dict['Combined Standard English + Combined Singlish + Nagation + Transport Domain + Too Handling + Like Handling + Adversative + Sarcasm'] = accuracy_score(df[Sentiment_Num], df["Polarity Count-after Adversative"])
pre_dict['Combined Standard English + Combined Singlish + Nagation + Transport Domain + Too Handling + Like Handling + Adversative + Sarcasm'] = precision_score(df[Sentiment_Num], df["Polarity Count-after Adversative"], average='weighted')
rec_dict['Combined Standard English + Combined Singlish + Nagation + Transport Domain + Too Handling + Like Handling + Adversative + Sarcasm'] = recall_score(df[Sentiment_Num], df["Polarity Count-after Adversative"], average='weighted')
f1_dict['Combined Standard English + Combined Singlish + Nagation + Transport Domain + Too Handling + Like Handling + Adversative + Sarcasm'] = f1_score(df[Sentiment_Num], df["Polarity Count-after Adversative"], average='weighted')

df_confusion = pd.crosstab(df['Polarity Count-after Adversative'],df["Sentiment Num"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
#df_confusion

sarcasm_df = df
#sarcasm_df = sarcasm_df.rename(columns={'Polarity Count-after Adversative': 'Sarcasm Polarity Count-after Adversative'}, replace=True)



'''
#Step 7: Combined Standard EL + Combined Singlish + Transport Domain + Negation + Too Handling + Sarcasm
df["Sarcasm?"] = df['Polarities Found'].apply(lambda row: recognise_sarcasm(row))
df.loc[(df['Sarcasm?'] != 0), 'Polarity Count'] = -1
acc_dict['Combined Standard English + Combined Singlish + Transport Domain + Nagation + Too Handling + Like Handling + Sarcasm'] = accuracy_score(df["Sentiment Num"], df["Polarity Count"])
f1_dict['Combined Standard English + Combined Singlish + Transport Domain + Nagation + Too Handling + Like Handling + Sarcasm'] = f1_score(df["Sentiment Num"], df["Polarity Count"], average='weighted')


df_confusion = pd.crosstab(df['Polarity Count'],df["Sentiment Num"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
#df_confusion

sarcasm_df = df
sarcasm_df = sarcasm_df.rename(columns={'Polarity Count': 'Sarcasm Polarity Count'})
'''

'''
#Step 8: Combined Standard EL + Combined Singlish + Transport Domain + Negation + Too Handling + Sarcasm + Adversative
df["Polarities Found"] = df['Text'].apply(lambda text: findPolarity5(' '.join(text.split())))
df["Adversative Polarity"] = df['Polarities Found'].apply(lambda scores: countPolarity5(scores, 7))
def adversative_present(polarity_list):
    if (8 in polarity_list) or (-8 in polarity_list):
        return 1
    else:
        return 0

# label presence of adversative
df['Adversative Present?']=df['Polarities Found'].apply(lambda pl:adversative_present(pl))


def update_p_after_adversative(present,polaritys, polaritya):
    if present==1:
        return polaritya
    elif present==0:
        return polaritys

df['Polarity Count-after Adversative'] = df.apply(lambda x: update_p_after_adversative(x['Adversative Present?'],x['Polarity Count'],x['Adversative Polarity']),axis=1)

df["Polarity Count-after Adversative"] = df.apply(lambda a: qn_mark(a['Text-original'],a['Polarity Count-after Adversative']),axis=1)
df['Polarity Count-after Adversative'].unique()
adversative_df = df

acc_dict['Combined Standard English + Combined Singlish + Transport Domain + Negation + Too Handling + Like Handling + Sarcasm + Adversative'] = accuracy_score(adversative_df["Sentiment Num"], adversative_df["Polarity Count-after Adversative"])
f1_dict['Combined Standard English + Combined Singlish + Transport Domain + Negation + Too Handling + Like Handling + Sarcasm + Adversative'] = f1_score(df["Sentiment Num"], df["Polarity Count"], average='weighted')

df_confusion = pd.crosstab(df['Polarity Count-after Adversative'],df["Sentiment Num"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
#df_confusion
'''


#Step 9: Combined Standard EL + Combined Singlish + Transport Domain + Negation + Too Handling + Adversative + Sarcasm + Emoji
adversative_df["Emoji Score"] = adversative_df['Text'].apply(lambda x: find_emoji(x))
adversative_df.loc[(adversative_df['Polarity Count-after Adversative'] == 0), 'Polarity Count-after Adversative'] = adversative_df['Emoji Score'] #emoji handling only when 0 is present
# for checking
adversative_df['Polarity Count-after Adversative'].unique()
acc_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain + Too Handling + Like Handling + Adversative  + Sarcasm + emoji'] = accuracy_score(adversative_df[Sentiment_Num], adversative_df["Polarity Count-after Adversative"])
pre_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain + Too Handling + Like Handling + Adversative + Sarcasm + emoji'] = precision_score(adversative_df[Sentiment_Num], adversative_df["Polarity Count"], average='weighted')
rec_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain + Too Handling + Like Handling + Adversative + Sarcasm + emoji'] = recall_score(adversative_df[Sentiment_Num], adversative_df["Polarity Count"], average='weighted')
f1_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain + Too Handling + Like Handling + Adversative + Sarcasm + emoji'] = f1_score(adversative_df[Sentiment_Num], adversative_df["Polarity Count"], average='weighted')

df_confusion = pd.crosstab(df['Polarity Count-after Adversative'],df["Sentiment Num"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
#df_confusion



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
adversative_df['Polarity Count-multi_final']=adversative_df['Polarity Count-multi'].apply(new_multi)

acc_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain  + Too Handling + Like Handling + Sarcasm + Adversative + Emoji + Multi'] = accuracy_score(adversative_df["Multi"], adversative_df["Polarity Count-multi_final"])
pre_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain  + Too Handling + Like Handling + Sarcasm + Adversative + Emoji + Multi'] = precision_score(adversative_df["Multi"], adversative_df["Polarity Count-multi_final"], average='weighted')
rec_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain  + Too Handling + Like Handling + Sarcasm + Adversative + Emoji + Multi'] = recall_score(adversative_df["Multi"], adversative_df["Polarity Count-multi_final"], average='weighted')
f1_dict['Combined Standard English + Combined Singlish + Negation + Transport Domain  + Too Handling + Like Handling + Sarcasm + Adversative + Emoji + Multi'] = f1_score(adversative_df["Multi"], adversative_df["Polarity Count-multi_final"], average='weighted')
'''
df_confusion = pd.crosstab(df['Polarity Count-multi'],df["Multi"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [2, 1, 0, -1, -2]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[2] + df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1] + df_confusion.loc[-2]
#df_confusion
'''
#adversative_df.to_csv('C:/Users/zhendahu/Desktop/adversative_df.csv')
adversative_df_part = adversative_df[adversative_df['Multi']!= adversative_df['Polarity Count-multi_final']]