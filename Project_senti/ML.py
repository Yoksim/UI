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
from sklearn.metrics import f1_score, accuracy_score

#data = pd.read_csv('C:/Users/zhendahu/Desktop/000workwell - to zhenda/validation with other existing models/0test-ready.csv')#sg-transport-1115-clean.csv')#sgtransport1145.csv')
data = pd.read_csv('C:/Users/zhendahu/Desktop/Project_senti/sg-transport-1115-clean.csv')
data_trainset = pd.read_csv('C:/Users/zhendahu/Desktop/Project_senti/0train-ready.csv')
data_testset = pd.read_csv('C:/Users/zhendahu/Desktop/Project_senti/0test-ready.csv')

# check data
print(data_trainset.shape)
data_trainset.head(10)


# basic pre-processing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(review):
    review = " ".join([stemmer.stem(w.lower()) for w in word_tokenize(re.sub('[^a-zA-Z]+', ' ', review.replace("<br />", ""))) if not w in stop_words])
    return review

#data['review_clean'] = data.apply(lambda x: preprocess(x['Text']), axis=1)
data_trainset['review_clean'] = data_trainset.apply(lambda x: preprocess(x['Text']), axis=1)
data_testset['review_clean'] = data_trainset.apply(lambda x: preprocess(x['Text']), axis=1)

#For transport
# split dataset to training and test dataset
'''
data_train, data_test, y_train, y_test = train_test_split(data['review_clean'], data['Multi'], 
                                                          test_size = 0.25, random_state = 2020, stratify = data['Multi'])
'''

#For Movie
data_train = data_trainset['review_clean']
data_test = data_testset['review_clean']
y_train = data_trainset['Sentiment Num']
y_test = data_testset['Sentiment Num']


vectorizer = CountVectorizer()
train_counts = vectorizer.fit_transform(data_train)
test_counts = vectorizer.transform(data_test)

#Model Fitting
lr = LogisticRegression()
lr.fit(train_counts, y_train)
lr_pred = lr.predict(test_counts)
print('Logistic Regression')
print('Accuracy:')
print(lr.score(test_counts, y_test))
print('Confusion Matrix:')
print(confusion_matrix(lr_pred, y_test))
print('F1 Score:')
print(f1_score(y_test, lr_pred, average='weighted'))


nb = MultinomialNB()
nb.fit(train_counts, y_train)
nb_pred = nb.predict(test_counts)
print('Naive Bayes')
print('Accuracy:')
print(nb.score(test_counts, y_test))
print('Confusion Matrix:')
print(confusion_matrix(nb_pred, y_test))
print('F1 Score:')
print(f1_score(y_test, nb_pred, average='weighted'))


start=datetime.now()
svm = LinearSVC()
svm.fit(train_counts, y_train)
svm_pred = svm.predict(test_counts)
print('SVM')
print('Accuracy:')
print(svm.score(test_counts, y_test))
print('Confusion Matrix:')
print(confusion_matrix(svm_pred, y_test))
print('F1 Score:')
print(f1_score(y_test, svm_pred, average='weighted'))


