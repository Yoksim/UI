# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 20:30:49 2021

@author: zhendahu
"""

from keras.models import Sequential
from keras.layers import Embedding, Activation, Dense, Dropout, SpatialDropout1D, LSTM, Bidirectional, Conv1D, MaxPooling1D, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import keras
import json
import pandas as pd
import numpy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score

data_trainset = pd.read_csv('C:/Users/zhendahu/Desktop/Project_senti/0train-ready.csv')
data_testset = pd.read_csv('C:/Users/zhendahu/Desktop/Project_senti/0test-ready-1.csv')



word2index = json.load(open('C:/Users/zhendahu/Desktop/Project_senti/doubleembedding/word_idx.json'))
general_embedding = numpy.load('C:/Users/zhendahu/Desktop/Project_senti/doubleembedding/gen.vec.npy')
#data = pd.concat([data_trainset, data_testset], axis=0, ignore_index = True)
data_review_train = data_trainset['Text']
data_review_test = data_testset['Text']

#data_review = [data_review[k].split(' ') for k in range(0, len(data_review))]
data_review_train = [data_review_train[k].split(' ') for k in range(0, len(data_review_train))]
data_review_test = [data_review_test[k].split(' ') for k in range(0, len(data_review_test))]
data_review = data_review_train + data_review_test
review_list = []
for review in data_review:
    review_list = review_list + review
vocabulary_review = list(set(review_list))
word_dictionary = {word: i+1 for i, word in enumerate(vocabulary_review)}
inverse_word_dictionary = {i+1: word for i, word in enumerate(vocabulary_review)}
vocab_size = len(word_dictionary.keys())


x_train = [[word_dictionary[word] for word in sent] for sent in data_review_train]
x_test = [[word_dictionary[word] for word in sent] for sent in data_review_test]
x_train = pad_sequences(maxlen=100, sequences=x_train, padding='post', value=0)
x_test = pad_sequences(maxlen=100, sequences=x_test, padding='post', value=0)
num_classes = 3
y_train =  data_trainset['Sentiment Num']
y_test =  data_testset['Sentiment Num']
y_train = np_utils.to_categorical(y_train,num_classes=num_classes) 
y_test = np_utils.to_categorical(y_test,num_classes=num_classes) 
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 2020, stratify = y)




embedding_matrix = numpy.zeros((vocab_size+1, 300))
for word, i in word_dictionary.items():
    try:
        embedding_vector = general_embedding[word2index[word]]
        embedding_matrix[i] = embedding_vector
    except:
        continue

#LSTM
lstm = Sequential()
lstm.add(Embedding(input_dim=vocab_size+1, output_dim=300, input_length=100, embeddings_initializer=keras.initializers.Constant(embedding_matrix), mask_zero=True))
#lstm.add(Embedding(input_dim=vocab_size+1, output_dim=300, input_length=100, mask_zero=True))
lstm.add(SpatialDropout1D(0.2))
lstm.add(LSTM(units=50, activation='relu'))
lstm.add(Dense(num_classes, activation='softmax'))
lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
lstm.fit(x_train, y_train, batch_size=16, epochs=20, verbose=1)
lstm_pred = lstm.predict(x_test)
test_sequences, pred_sequences = [], []
for i in range(len(lstm_pred)):
    pred_sequences.append(numpy.argmax(lstm_pred[i]))#返回预测标签索引
    test_sequences.append(numpy.argmax(y_test[i]))#返回真实标签索引
print('LSTM:')
print('Accuracy:')
print(accuracy_score(numpy.array(pred_sequences), numpy.array(test_sequences)))
print('Confusion Matrix:')
print(confusion_matrix(numpy.array(pred_sequences), numpy.array(test_sequences)))
print('F1 Score:')
print(f1_score(numpy.array(pred_sequences), numpy.array(test_sequences), average='weighted'))




#CNN
cnn = Sequential()
cnn.add(Embedding(input_dim=vocab_size+1, output_dim=300, input_length=100, embeddings_initializer=keras.initializers.Constant(embedding_matrix), mask_zero=True))
#cnn.add(Embedding(input_dim=vocab_size+1, output_dim=300, input_length=100, mask_zero=True))
cnn.add(SpatialDropout1D(0.2))
cnn.add(Conv1D(32, 5, padding = 'same', activation='relu'))
cnn.add(MaxPooling1D(3, 3, padding = 'same'))
cnn.add(Dropout(0.2))
cnn.add(Flatten())
cnn.add(Dense(num_classes, activation='softmax'))
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.fit(x_train, y_train, batch_size=16, epochs=20, verbose=1)
cnn_pred = cnn.predict(x_test)
test_sequences, pred_sequences = [], []
for i in range(len(lstm_pred)):
    pred_sequences.append(numpy.argmax(cnn_pred[i]))#返回预测标签索引
    test_sequences.append(numpy.argmax(y_test[i]))#返回真实标签索引
print('CNN:')
print('Accuracy:')
print(accuracy_score(numpy.array(pred_sequences), numpy.array(test_sequences)))
print('Confusion Matrix:')
print(confusion_matrix(numpy.array(pred_sequences), numpy.array(test_sequences)))
print('F1 Score:')
print(f1_score(numpy.array(pred_sequences), numpy.array(test_sequences), average='weighted'))


