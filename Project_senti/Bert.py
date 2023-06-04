# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 21:32:19 2021

@author: zhendahu
"""

import random
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

device = torch.device('cpu')


# Get text values and labels
#data = pd.read_csv('C:/Users/zhendahu/Desktop/Project_senti/sg-transport-1115-clean.csv')
#data = pd.read_csv('C:/Users/zhendahu/Desktop/Project_senti/0test-ready-1.csv')
#data = pd.read_csv('C:/Users/zhendahu/Desktop/Project_senti/new_raw_movie.csv')
data = pd.read_csv('C:/Users/zhendahu/Desktop/Data/data_part.csv')

text_values = data['Text'].values
labels_pre = data['Multi'].values
#labels_pre = data['Sentiment Num'].values
num_labels = 5
#labels = np_utils.to_categorical(labels_pre,num_classes=num_labels) 

# Load the pretrained Tokenizer
tokenizer = BertTokenizer.from_pretrained('C:/Users/zhendahu/Desktop/Project_senti/bert-base-uncased/bert-base-uncased-vocab.txt', do_lower_case=True)

print('Original Text : ', text_values[1])
print('Tokenized Text: ', tokenizer.tokenize(text_values[1]))
print('Token IDs     : ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_values[1])))


# Function to get token ids for a list of texts 
def encode_fn(text_list):
    all_input_ids = []    
    for text in text_list:
        input_ids = tokenizer.encode(
                        text,                      
                        add_special_tokens = True,  # 添加special tokens， 也就是CLS和SEP
                        max_length = 100,           # 设定最大文本长度
                        pad_to_max_length = True,   # pad到最大的长度  
                        return_tensors = 'pt'       # 返回的类型为pytorch tensor
                   )
        all_input_ids.append(input_ids)    
    all_input_ids = torch.cat(all_input_ids, dim=0)
    return all_input_ids

all_input_ids = encode_fn(text_values)
#labels = torch.tensor(labels)

epochs = 1
batch_size = 16

# Split data into train and validation
#dataset = TensorDataset(all_input_ids, labels)

'''
train_size = int(0.75 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
'''

#KFold
skf = StratifiedKFold(n_splits = 4, random_state=2020, shuffle=True)
skf.get_n_splits()
for train_index, test_index in skf.split(text_values, labels_pre):
    x_train_dataset, x_val_dataset = all_input_ids[train_index], all_input_ids[test_index]
    y_train_dataset, y_val_dataset = labels_pre[train_index], labels_pre[test_index]
    
    y_train_dataset = np_utils.to_categorical(y_train_dataset,num_classes=num_labels) 
    y_val_dataset = np_utils.to_categorical(y_val_dataset,num_classes=num_labels) 
    
    y_train_dataset = torch.tensor(y_train_dataset)
    y_val_dataset = torch.tensor(y_val_dataset)
    
    train_dataset = TensorDataset(x_train_dataset, y_train_dataset)
    val_dataset = TensorDataset(x_val_dataset, y_val_dataset)
    
    # Create train and validation dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_dataloader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False)
    
    # Load the pretrained BERT model
    model = BertForSequenceClassification.from_pretrained('C:/Users/zhendahu/Desktop/Project_senti/bert-base-uncased/', num_labels=num_labels, output_attentions=False, output_hidden_states=False)
    #model.cuda()
    
    # create optimizer and learning rate schedule
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    def flat_accuracy(preds, labels):
        
        """A function for calculating accuracy scores"""
        
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = np.argmax(labels, axis=1).flatten()
        return accuracy_score(labels_flat, pred_flat)
    
    
    
    for epoch in range(epochs):
        model.train()
        total_loss, total_val_loss = 0, 0
        total_eval_accuracy = 0
        for step, batch in enumerate(train_dataloader):
            model.zero_grad()
            output = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0]>0).to(device), labels=batch[1].to(device))
            loss = output.loss
            logits = output.logits
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step() 
            scheduler.step()
            
        model.eval()
        for i, batch in enumerate(val_dataloader):
            with torch.no_grad():
                output = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0]>0).to(device), labels=batch[1].to(device))
                loss = output.loss
                logits = output.logits
                total_val_loss += loss.item()
                
                logits = logits.detach().cpu().numpy()
                label_ids = batch[1].to('cpu').numpy()
                total_eval_accuracy += flat_accuracy(label_ids, logits)
        
        avg_train_loss = total_loss / len(train_dataloader)
        avg_val_loss = total_val_loss / len(val_dataloader)
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        
        print(f'Train loss: {avg_train_loss}')
        print(f'Validation loss: {avg_val_loss}')
        print(f'Accuracy: {avg_val_accuracy:.2f}')
        print('\n')
    
    
    # Create the test data loader
    pred_dataloader = val_dataloader
    
    model.eval()
    preds = []
    true_labels = []
    for i, batch in enumerate(pred_dataloader):
        with torch.no_grad():
            outputs = model(batch[0].to(device), token_type_ids=None, attention_mask=(batch[0]>0).to(device))
            logits = outputs.logits
            logits = logits.detach().cpu().numpy()
            preds.append(logits)
            labels = batch[1]
            true_labels.append(labels)
    
    final_preds = np.concatenate(preds, axis=0)
    final_preds = np.argmax(final_preds, axis=1)
    final_labels = np.concatenate(true_labels, axis=0)
    final_labels = np.argmax(final_labels, axis=1)
    
    print('BERT:')
    print('Accuracy:')
    print(accuracy_score(final_labels, final_preds))
    print('Confusion Matrix:')
    print(confusion_matrix(final_labels, final_preds))
    print('Precision Score:')
    print(precision_score(final_labels, final_preds, average='weighted'))
    print('Recall Score:')
    print(recall_score(final_labels, final_preds, average='weighted'))
    print('F1 Score:')
    print(f1_score(final_labels, final_preds, average='weighted'))

