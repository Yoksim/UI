# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 19:28:20 2021

@author: zhendahu
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 19:22:48 2021

@author: zhendahu
"""

# import data
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re
from datetime import datetime
begin = datetime.now()
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from keras.utils import np_utils

#data = pd.read_csv('C:/Users/zhendahu/Desktop/000workwell - to zhenda/validation with other existing models/0test-ready.csv')#sg-transport-1115-clean.csv')#sgtransport1145.csv')
#data = pd.read_csv('C:/Users/zhendahu/Desktop/Project_senti/sg-transport-1115-clean.csv')
#data = pd.read_csv('C:/Users/zhendahu/Desktop/Project_senti/0test-ready-1.csv')
data = pd.read_csv('C:/Users/zhendahu/Desktop/Project_senti/new_raw_movie.csv')
data = pd.read_csv('C:/Users/zhendahu/Desktop/Data/data_part.csv')
data = pd.read_excel('C:/Users/zhendahu/Desktop/Data/200Han.xlsx')


# basic pre-processing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(review):
    review = " ".join([stemmer.stem(w.lower()) for w in word_tokenize(re.sub('[^a-zA-Z]+', ' ', review.replace("<br />", ""))) if not w in stop_words])
    return review

data['review_clean'] = data.apply(lambda x: preprocess(x['Text']), axis=1)

#For transport
# split dataset to training and test dataset
'''
data_train, data_test, y_train, y_test = train_test_split(data['review_clean'], data['Multi'], 
                                                          test_size = 0.25, random_state = 2020, stratify = data['Multi'])
'''

#For Movie
x = data['review_clean']
y = data['Sentiment Num']
#y = data['Multi']

#word frequency matrix
vectorizer = CountVectorizer()
x_counts = vectorizer.fit_transform(x).toarray()
y_counts = y.values



#KFold
skf = StratifiedKFold(n_splits = 4, random_state=2020, shuffle=True)
skf.get_n_splits()
for train_index, test_index in skf.split(x_counts, y_counts):
    x_train, x_test = x_counts[train_index], x_counts[test_index]
    y_train, y_test = y_counts[train_index], y_counts[test_index]


    #Model Fitting
    lr = LogisticRegression()
    lr.fit(x_train, y_train)
    lr_pred = lr.predict(x_test)
    print('Logistic Regression')
    print('Accuracy:')
    print(accuracy_score(y_test, lr_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, lr_pred))
    print('Precision Score:')
    print(precision_score(y_test, lr_pred, average='weighted'))
    print('Recall Score:')
    print(recall_score(y_test, lr_pred , average='weighted'))
    print('F1 Score:')
    print(f1_score(y_test, lr_pred, average='weighted'))


    nb = MultinomialNB()
    nb.fit(x_train, y_train)
    nb_pred = nb.predict(x_test)
    print('Naive Bayes')
    print('Accuracy:')
    print(accuracy_score(y_test, nb_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, nb_pred))
    print('Precision Score:')
    print(precision_score(y_test, nb_pred, average='weighted'))
    print('Recall Score:')
    print(recall_score(y_test, nb_pred , average='weighted'))
    print('F1 Score:')
    print(f1_score(y_test, nb_pred, average='weighted'))


    start=datetime.now()
    svm = LinearSVC()
    svm.fit(x_train, y_train)
    svm_pred = svm.predict(x_test)
    print('SVM')
    print('Accuracy:')
    print(accuracy_score(y_test, svm_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, svm_pred))
    print('Precision Score:')
    print(precision_score(y_test, svm_pred, average='weighted'))
    print('Recall Score:')
    print(recall_score(y_test, svm_pred , average='weighted'))
    print('F1 Score:')
    print(f1_score(y_test, svm_pred, average='weighted'))


