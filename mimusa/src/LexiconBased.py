#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
from sklearn.metrics import accuracy_score


# In[24]:


from sentiment_explorerVersion7 import *


# In[25]:


def pdColumn_to_list_converter(df):
    df_list = df.values.tolist() #produces list of lists
    proper_list = [item for sublist in df_list for item in sublist] #a single list
    return proper_list


# In[26]:


#df = pd.read_csv('amplifier-test---example.csv')  #import reviews
df = pd.read_csv('sg-transport-1115-clean.csv')  #import reviews
#df = pd.read_csv('Negate_and_negation_test_example.csv')  #import reviews
#df = pd.read_csv('initial data as well as targeted output-RA-data-example -jiaxin-simple.csv')  #import reviews
#df = pd.read_csv('0train-ready.csv')  #import reviews
#df = pd.read_csv('0test-ready.csv')  #import reviews
df.head()


# In[27]:


df['Text-original'] = df['Text']
df["Text"] = df['Text'].apply(lambda text: text.lower())
df.head()


# In[28]:


df_replacement_words = pd.read_csv('ngram_20170222_ZX2021_replace_words.csv')   ##import replace words ###ngram_20170222_ZX2021_replace_words
df_replacement_words["original_phases"] = df_replacement_words["original_phases"].apply(lambda text: text.lower())
df_replacement_words["target_produced_words1"] = df_replacement_words["target_produced_words1"].apply(lambda text: text.lower())
df_replacement_words["target_produced_words2"] = df_replacement_words["target_produced_words2"].apply(lambda text: text.lower())
df_replacement_words.head()


# In[29]:


replace_words_dic = df_replacement_words.set_index('original_phases').to_dict()['target_produced_words1'] ## create a dic for replacing words
replace_words_dic


# ### Removing Punctuations From Data

# In[30]:


import re

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


# In[32]:


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


# In[33]:


df["Text_with_fullstop"] = df['Text'].apply(lambda text: newtext_fullstop(text))


# In[34]:


df["Text"] = df['Text'].apply(lambda text: newtext(text))


# In[35]:


df.drop(df.columns[0], axis=1)


# ### Counting accuracy

# In[36]:


acc_dict = {}


# ## Step 1: Using Prof Wang's Standard English Dictionary Only

# In[37]:


df["Polarities Found"] = df['Text'].apply(lambda text: findPolarity(' '.join(text.split())))


# In[38]:


df["Polarity Count"] = df['Polarities Found'].apply(lambda scores: countPolarity(scores))


# In[39]:


acc_dict['Prof Wang Standard English Only'] = accuracy_score(df["Sentiment Num"], df["Polarity Count"])


# In[40]:


acc_dict


# In[41]:


df_confusion = pd.crosstab(df['Polarity Count'],df["Sentiment Num"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
df_confusion


# In[42]:


df = df.drop(df.columns[-2:], axis=1)


# ## Step2 : Using Prof Wang's Standard el + Prof Wang Singlish Dict 

# In[43]:


df["Polarities Found"] = df['Text'].apply(lambda text: findPolarity1(' '.join(text.split())))


# In[44]:


df["Polarity Count"] = df['Polarities Found'].apply(lambda scores: countPolarity1(scores))


# In[45]:


acc_dict['Prof Wang Standard English + Prof Wang Singlish'] = accuracy_score(df["Sentiment Num"], df["Polarity Count"])


# In[46]:



acc_dict


# In[47]:


df_confusion = pd.crosstab(df['Polarity Count'],df["Sentiment Num"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
df_confusion


# In[48]:


df = df.drop(df.columns[-2:], axis=1)


# # Step 3: Combined Standard EL + Combined Singlish

# In[49]:


df["Polarities Found"] = df['Text'].apply(lambda text: findPolarity2(' '.join(text.split())))


# In[50]:


df["Polarity Count"] = df['Polarities Found'].apply(lambda scores: countPolarity2(scores))


# In[51]:


acc_dict['Combined Standard English + Combined Singlish'] = accuracy_score(df["Sentiment Num"], df["Polarity Count"])


# In[52]:


acc_dict


# In[53]:


df_confusion = pd.crosstab(df['Polarity Count'],df["Sentiment Num"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
df_confusion


# In[31]:


df = df.drop(df.columns[-2:], axis=1)


# In[32]:


df


# # Step 4: Combined Standard EL + Combined Singlish + Transport Domain

# In[33]:


df["Polarities Found"] = df['Text'].apply(lambda text: findPolarity3(' '.join(text.split())))


# In[34]:


df["Polarity Count"] = df['Polarities Found'].apply(lambda scores: countPolarity3(scores))


# In[35]:


acc_dict['Combined Standard English + Combined Singlish + Transport Domain'] = accuracy_score(df["Sentiment Num"], df["Polarity Count"])


# In[36]:


acc_dict


# In[37]:


df


# In[38]:


df_confusion = pd.crosstab(df['Polarity Count'],df["Sentiment Num"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
df_confusion


# In[39]:


df = df.drop(df.columns[-2:], axis=1)


# In[40]:


# df = df.drop(['Polarities Found'], axis=1)


# In[41]:


# df = df.rename(columns={"Polarity Count": "Polarity Count after Transport Domain"})


# # Step 5: Combined Standard EL + Combined Singlish + Transport Domain + Negation

# In[42]:


df["Polarities Found"] = df['Text_with_fullstop'].apply(lambda text: findPolarity4(' '.join(text.split())))


# In[43]:


df["Polarity Count"] = df['Polarities Found'].apply(lambda scores: countPolarity4(scores, 7))


# In[44]:


acc_dict['Combined Standard English + Combined Singlish + Transport Domain + Negation'] = accuracy_score(df["Sentiment Num"], df["Polarity Count"])


# In[45]:


acc_dict


# In[46]:


df_confusion = pd.crosstab(df['Polarity Count'],df["Sentiment Num"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
df_confusion


# In[47]:


df


# In[ ]:





# In[48]:


# df.to_csv("checking.csv")


# In[49]:


df = df.drop(df.columns[-2:], axis=1)


# In[50]:


# df = df.rename(columns={"Polarity Count": "Polarity Count after Negation"})


# In[51]:


# df_wrongly_labelled_negation = df[df['Polarity Count after Negation'] != df['Sentiment Num']]


# In[52]:


# df_wrongly_labelled_negation.to_csv('wrongly_labelled_negation.csv')


# In[53]:


# df_wrongly_labelled_negation_vs_transport = df_wrongly_labelled_negation[df_wrongly_labelled_negation['Polarity Count after Negation'] != df_wrongly_labelled_negation['Polarity Count after Transport Domain']]


# In[54]:


# df_wrongly_labelled_negation_vs_transport


# In[55]:


# df_wrongly_labelled_negation_vs_transport_correct = df_wrongly_labelled_negation_vs_transport[df_wrongly_labelled_negation_vs_transport['Sentiment Num'] == df_wrongly_labelled_negation_vs_transport['Polarity Count after Transport Domain']]


# In[56]:


# df_wrongly_labelled_negation_vs_transport_correct


# In[57]:


# df_wrongly_labelled_negation_vs_transport_correct.to_csv('transport_correct_label_negation_wrong.csv')


# # Step 6: Combined Standard EL + Combined Singlish + Transport Domain + Negation + Too Handling

# In[58]:


df["Polarities Found"] = df['Text'].apply(lambda text: findPolarity4_too(' '.join(text.split())))


# In[59]:


df["Polarity Count"] = df['Polarities Found'].apply(lambda scores: countPolarity4(scores, 7))


# In[60]:


acc_dict['Combined Standard English + Combined Singlish + Transport Domain + Negation + Too Handling'] = accuracy_score(df["Sentiment Num"], df["Polarity Count"])


# In[61]:


acc_dict


# In[62]:


df_confusion = pd.crosstab(df['Polarity Count'],df["Sentiment Num"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
df_confusion


# # Step 6.1: Combined Standard EL + Combined Singlish + Transport Domain + Negation + Too Handling + Like Handling

# In[63]:


df["Polarities Found"] = df['Text'].apply(lambda text: findPolarity4_too_like(' '.join(text.split())))


# In[64]:


df["Polarity Count"] = df['Polarities Found'].apply(lambda scores: countPolarity4(scores, 7))


# In[65]:


acc_dict['Combined Standard English + Combined Singlish + Transport Domain + Negation + Too Handling + Like Handling'] = accuracy_score(df["Sentiment Num"], df["Polarity Count"])


# In[66]:


acc_dict


# In[67]:


df_confusion = pd.crosstab(df['Polarity Count'],df["Sentiment Num"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
df_confusion


# # Step 6.2: Combined Standard EL + Combined Singlish + Transport Domain + Negation + Too Handling + Like Handling + Question mark Handling

# In[68]:


#df.head()


# In[69]:


def qn_mark(original_text, polarity):
    fivewoneh=['what','why','who','where','when','how', 'What','Why','Who','Where','When','How']
    if '?' in original_text:
        original_text = original_text.strip()
        if original_text.split(" ")[0] not in fivewoneh:
            polarity=-1
    else:
        polarity=polarity
    return polarity
        


# In[70]:


df["Polarity Count"] = df.apply(lambda a: qn_mark(a['Text-original'],a['Polarity Count']),axis=1)


# In[71]:


#df.head()


# In[72]:


acc_dict['Combined Standard English + Combined Singlish + Transport Domain + Negation + Too Handling + Like Handling + Qn Mark Handling'] = accuracy_score(df["Sentiment Num"], df["Polarity Count"])


# In[73]:


acc_dict


# In[74]:


df_confusion = pd.crosstab(df['Polarity Count'],df["Sentiment Num"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
df_confusion


# # Step 7: Combined Standard EL + Combined Singlish + Transport Domain + Negation + Too Handling + Sarcasm

# In[75]:


df["Sarcasm?"] = df['Polarities Found'].apply(lambda row: recognise_sarcasm(row))


# In[76]:



df.loc[(df['Sarcasm?'] != 0), 'Polarity Count'] = -1


# In[77]:


acc_dict['Combined Standard English + Combined Singlish + Transport Domain + Nagation + Too Handling + Like Handling + Sarcasm'] = accuracy_score(df["Sentiment Num"], df["Polarity Count"])


# In[78]:


acc_dict


# In[79]:


df_confusion = pd.crosstab(df['Polarity Count'],df["Sentiment Num"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
df_confusion


# In[80]:


sarcasm_df = df


# In[81]:


sarcasm_df


# In[82]:


sarcasm_df = sarcasm_df.rename(columns={'Polarity Count': 'Sarcasm Polarity Count'})


# In[83]:


sarcasm_df


# # Step 8: Combined Standard EL + Combined Singlish + Transport Domain + Negation + Too Handling + Sarcasm + Adversative

# In[84]:



df["Polarities Found"] = df['Text'].apply(lambda text: findPolarity5(' '.join(text.split())))
df["Adversative Polarity"] = df['Polarities Found'].apply(lambda scores: countPolarity5(scores, 7))


# In[85]:


# def flip(sarcasm,polarity_count):
#     if sarcasm==-1:
#         if polarity_count>0:
#             return polarity_count*(-1)
#         else:
#             return polarity_count
#     elif sarcasm==0:
#         return polarity_count


# In[86]:


def adversative_present(polarity_list):
    if (8 in polarity_list) or (-8 in polarity_list):
        return 1
    else:
        return 0

# label presence of adversative
df['Adversative Present?']=df['Polarities Found'].apply(lambda pl:adversative_present(pl))


# In[87]:


def update_p_after_adversative(present,polaritys, polaritya):
    if present==1:
        return polaritya
    elif present==0:
        return polaritys

df['Polarity Count-after Adversative'] = df.apply(lambda x: update_p_after_adversative(x['Adversative Present?'],x['Polarity Count'],x['Adversative Polarity']),axis=1)


# In[88]:


df["Polarity Count-after Adversative"] = df.apply(lambda a: qn_mark(a['Text-original'],a['Polarity Count-after Adversative']),axis=1)


# In[89]:


df


# In[90]:


# checking
df['Polarity Count-after Adversative'].unique()


# In[91]:


adversative_df = df


# In[92]:


adversative_df


# In[93]:


acc_dict['Combined Standard English + Combined Singlish + Transport Domain + Negation + Too Handling + Like Handling + Sarcasm + Adversative'] = accuracy_score(adversative_df["Sentiment Num"], adversative_df["Polarity Count-after Adversative"])


# In[94]:


acc_dict


# In[95]:


df_confusion = pd.crosstab(df['Polarity Count-after Adversative'],df["Sentiment Num"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
df_confusion


# In[ ]:





# # Step 9: Combined Standard EL + Combined Singlish + Transport Domain + Negation + Too Handling + Sarcasm + Adversative +  Emoji

# In[96]:


adversative_df


# In[97]:


adversative_df["Emoji Score"] = adversative_df['Text'].apply(lambda x: find_emoji(x))
adversative_df.loc[(adversative_df['Polarity Count-after Adversative'] == 0), 'Polarity Count-after Adversative'] = adversative_df['Emoji Score'] #emoji handling only when 0 is present


# In[98]:


adversative_df


# In[99]:


# for checking
adversative_df['Polarity Count-after Adversative'].unique()


# In[100]:


acc_dict['Combined Standard English + Combined Singlish + Transport Domain + Negation + Too Handling + Like Handling + Sarcasm + Adversative+ emoji'] = accuracy_score(adversative_df["Sentiment Num"], adversative_df["Polarity Count-after Adversative"])


# In[101]:


acc_dict


# In[102]:


df_confusion = pd.crosstab(df['Polarity Count-after Adversative'],df["Sentiment Num"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]
df_confusion



# # Step 10: Combined Standard EL + Combined Singlish + Transport Domain + Negation + Too Handling + Sarcasm + Adversative + Emoji + Multi

# In[103]:


adversative_df


# In[104]:


adversative_df["Polarities Found"] = adversative_df['Text'].apply(lambda text: findPolarity6(' '.join(text.split())))


# In[105]:


adversative_df['Polarity Count-multi'] = adversative_df.apply(lambda scores: multi_value(scores['Polarities Found'],scores['Text'], scores['Polarity Count-after Adversative'], 5), axis=1)


# In[106]:


adversative_df["Polarity Count-multi"] = adversative_df.apply(lambda a: qn_mark(a['Text-original'],a['Polarity Count-multi']),axis=1)


# In[107]:


adversative_df # the last column gives us more information on the strength polarity


# In[108]:


# for checking
adversative_df['Polarity Count-multi'].unique()


# In[109]:


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

adversative_df['Polarity Count-multi']=adversative_df['Polarity Count-multi'].apply(new_multi)
# adversative_df['Multi']=adversative_df['Multi'].apply(new_multi)


# In[110]:



acc_dict['Combined Standard English + Combined Singlish + Transport Domain + Negation + Too Handling + Like Handling + Sarcasm + Adversative+ Emoji + Multi'] = accuracy_score(adversative_df["Multi"], adversative_df["Polarity Count-multi"])


# In[111]:


acc_dict


# In[112]:


df_confusion = pd.crosstab(df['Polarity Count-multi'],df["Multi"] , rownames=['Predicted'], colnames=['Actual'], margins= True)
labels = [2, 1, 0, -1, -2]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
df_confusion.loc['All'] = df_confusion.loc[2] + df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1] + df_confusion.loc[-2]
df_confusion


# In[113]:


adversative_df.head()


# In[114]:


# for checking
adversative_df['Polarity Count-multi'].unique()


# In[115]:


# uncomment the following for checking
adversative_df.to_csv('Results-lexiconbased-1115-1.csv')


# # Step 11: Evaluation

# In[116]:


# for adversative since multi is in another ipynb
df_confusion = pd.crosstab(adversative_df['Sentiment Num'],adversative_df["Polarity Count-after Adversative"] , rownames=['Actual'], colnames=['Predicted'], margins=True)
labels = [1, 0, -1]
df_confusion = df_confusion.reindex(labels, axis="columns")
df_confusion = df_confusion.reindex(labels, axis="rows")
# df_confusion.loc['All'] = df_confusion.loc[1] + df_confusion.loc[0] + df_confusion.loc[-1]


# In[117]:


df_confusion


# In[118]:


# normalised confusion matrix
df_conf_norm = df_confusion / df_confusion.sum(axis=1)
df_conf_norm


# In[119]:


# confusion matrix plot
import matplotlib.pyplot as plt
import numpy as np

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

plot_confusion_matrix(df_confusion)


# In[120]:


# plot normalized confusion matrix

plot_confusion_matrix(df_conf_norm)  


# In[121]:


# f1 score
from sklearn.metrics import f1_score
f1_score(adversative_df['Sentiment Num'], adversative_df['Polarity Count-after Adversative'], average='weighted')


# In[122]:


f1_score(adversative_df['Sentiment Num'], adversative_df['Polarity Count-after Adversative'], average='micro')


# In[123]:


f1_score(adversative_df['Sentiment Num'], adversative_df['Polarity Count-after Adversative'], average='macro')


# ##  Step 12: Cross Validation

# In[124]:


# # shuffle the data

# df = df.sample(frac=1).reset_index(drop=True)
# print(df)


# In[125]:


# df1=df[:229]
# print(df1)
# df1.to_csv('df1.csv')


# In[126]:


# df2=df[229:458]
# print(df2)
# df2.to_csv('df2.csv')


# In[127]:


# df3=df[458:687]
# print(df3)
# df3.to_csv('df3.csv')


# In[128]:


# df4=df[687:916]
# print(df4)
# df4.to_csv('df4.csv')


# In[129]:


# df5=df[916:]
# print(df5)
# df5.to_csv('df5.csv')


# In[130]:


# rerun the model on each df to test if the accuracy is stable


# # Step 13: Majority Voting

# In[131]:


print(df.head())


# In[132]:


df.groupby(by='Sentiment').count()


# In[133]:


df['allnegative']=-1


# In[134]:


print(df.head())


# In[135]:


from sklearn.metrics import accuracy_score
# supposed our model predict all neutral, benchmark
print(accuracy_score(df["Sentiment Num"], df["allnegative"]))


# In[ ]:




