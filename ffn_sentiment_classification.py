# -*- coding: utf-8 -*-
"""
   Assignment 3: Sentiment Classification on a Feed-Forward Neural Network using Pretrained Embeddings
   Original code by Hande Celikkanat & Miikka Silfverberg.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gensim
import os
import csv

# Add the path to these data manipulation files if necessary:
# import sys
# sys.path.append('</PATH/TO/DATA/MANIP/FILES>')
from data_semeval import *
from paths import data_dir, model_dir

# name of the embeddings file to use
# Alternatively, you can also use the full set,
#
# GoogleNews-vectors-negative300.bin (from https://code.google.com/archive/p/word2vec/)
embeddings_file = 'GoogleNews-pruned2tweets.bin'

# --- hyperparameters ---

n_classes = len(LABEL_INDICES)
n_epochs = 10                   # n of epochs didn't seem to have a big impact on the perfomance of the model
learning_rate = 0.001
report_every = 1   
verbose = False   


# --- auxilary functions ---

#   string label to pytorch tensor:
def label_to_idx(label):
    return torch.LongTensor([LABEL_INDICES[label]])

#   pytorch tensor back to a string label:
def idx_to_label(label):
    negative = label_to_idx('negative')
    neutral = label_to_idx('neutral')
    positive = label_to_idx('positive')
    idx_label = {negative.item() : "negative", neutral.item() : "neutral", positive.item(): "positive"}
    return idx_label[label.item()]


#   the tweets into lists of indices:    
def tweet_into_indices_and_embeddings(tweet):
    """
    Args: a tweet, as a list of tokens

    Creates a list of token indices retrieved from 'word_to_idx'.
    Then, retrieves the pretrained embeddings for these tokens from 'pretrained embeds' with the help of the indices from 
    'token_indices_in_tweet'. Creates a stacked tensor of these token embeddings.

    Returns: a 1x300 tensor, representing a tweet (token embeddings summed).

    """
    token_indices_in_tweet = []    #will be a list holding the token indices of a tweet
    token_embeddings = []
    for token in tweet['BODY']:
        try:
            tokenidx = word_to_idx[token]  #retrieving the index from 
            token_indices_in_tweet.append(tokenidx)
        except KeyError:   # there was a KeyError with value '3.39'??
            pass
    for token_idx in token_indices_in_tweet:
        embedding = torch.tensor(pretrained_embeds[token_idx]) #retrieving the embedding from the pretrained_embeds
        token_embeddings.append(embedding)
    stacked_embeddings = torch.stack(token_embeddings, dim=0)
    final_tweet_embedding = torch.sum(stacked_embeddings, dim=0)
    return final_tweet_embedding

# --- model ---

class FFNN(nn.Module):
    """ This Feed-forward model has two fully connected layers, fc1 and fc2.
        The output dimension of the first layer/input dimension of the second layer is 100. 
        Activation function used: relu.
        Sotfmax used: log_softmax
        Returns: a probabalitity distribution tensor of the labels.
          """

    def __init__(self, pretrained_embeds, n_classes, extra_arg_1=None, extra_arg_2=None):
        super(FFNN, self).__init__()

        hidden_dim = 100
        self.fc1 = nn.Linear(300, hidden_dim) # 300 since it is the vector dimensionality of a word.
        self.fc2 = nn.Linear(hidden_dim, 3)
        
        # initializing our activation function (relu) and softmax
        self.activationfunction = F.relu
        self.softmax = F.log_softmax
        

    def forward(self, x):
        tweet_embedding = tweet_into_indices_and_embeddings(x) # calling our embedding-making function here
        intermediate1 = self.fc1(tweet_embedding)
        intermediate2 = self.activationfunction(intermediate1)
        intermediate3 = self.fc2(intermediate2)
        output = self.softmax(intermediate3, dim=0)
        return output               

# --- "main" ---

if __name__ == '__main__':

    # --- data loading ---
    data = read_semeval_datasets(data_dir)
    gensim_embeds = gensim.models.KeyedVectors.load_word2vec_format(os.path.join(model_dir, embeddings_file),
                                                                    binary=True)    #vocabulary size: 14835, dimensionality of the vectors: 300
    pretrained_embeds = gensim_embeds.vectors  #transforms into a Numpy array (nested lists)

    # To convert words in the input tweet to indices of the embeddings matrix:
    word_to_idx = {}
    for i, word in enumerate(gensim_embeds.index_to_key):    #  index_to_key is all the words in the vocabulary
        word_to_idx[word] = i

    # --- set up ---

    model = FFNN(pretrained_embeds, n_classes) #initializing the model
    loss_function = torch.nn.NLLLoss()   
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    #--- training ---
    for epoch in range(n_epochs):
        raw_data = data['training']
        total_loss = 0
        for tweet in raw_data:
            gold_class = label_to_idx(tweet['SENTIMENT'])
            optimizer.zero_grad()
            y_pred = model(tweet)   
            #y_pred.requires_grad = True    --> I guess we don't need this? Is the grad settin already in as default?
            loss = loss_function(y_pred.unsqueeze(0), gold_class)
            total_loss = total_loss + loss.item()
            loss.backward()
            optimizer.step()

        if ((epoch + 1) % report_every) == 0:
            print(f"epoch: {epoch}, loss: {round(total_loss * 100 / len(data['training']), 4)}")
            
    #   Defining a fuction for writing our model's label predictions to test.input.txt
    def write_label_predictions(source_filepath, new_filepath, pred_labels):
        with open (source_filepath, encoding="utf-8") as sourcefile:
            sourcefile_reader = csv.reader(sourcefile, delimiter="\t")
            sourcefile_data = list(sourcefile_reader)
            
            for i, row in enumerate(sourcefile_data):
                row[1] = pred_labels[i]                 #   replacing the "unknown" tag with the predicted labels here

        with open (new_filepath, 'w', encoding="utf-8") as targetfile:   #writing our labels in the new file
            target_writer = csv.writer(targetfile, delimiter="\t")
            target_writer.writerows(sourcefile_data)


    # --- test with dev ---
    correct_dev = 0
    with torch.no_grad():
        for tweet in data['dev.gold']:
            gold_class_dev = label_to_idx(tweet['SENTIMENT'])
            y_pred_dev = np.argmax(model(tweet))
            correct_dev += torch.eq(y_pred_dev, gold_class_dev).item()
            if verbose:
                print('DEV DATA: %s, OUTPUT: %s, GOLD LABEL: %d' %
                      (tweet['BODY'], tweet['SENTIMENT'], y_pred_dev))

        print(f"dev accuracy: {round(100 * correct_dev / len(data['dev.gold']), 2)}")

# --- test with test set ---
    correct_test = 0
    test_predictions = []
    with torch.no_grad():
        for tweet_input, tweet_gold in zip(data['test.input'],data['test.gold']):
            y_pred_test = np.argmax(model(tweet_input))
            test_predictions.append(idx_to_label(y_pred_test))   #transforming the label back to a string now

            gold_class_test = label_to_idx(tweet_gold['SENTIMENT'])

            correct_test += torch.eq(y_pred_test, gold_class_test).item()

            if verbose:
                print('TEST DATA: %s, OUTPUT: %s, GOLD LABEL: %d' %
                      (tweet['BODY'], tweet['SENTIMENT'], y_pred_test))
                
        print(f"test accuracy: {round(100 * correct_test / len(data['test.gold']), 2)}")

        #calling the writing labels function here. Test.input.txt becomes --> assign3_FFNN_lr01.txt
        write_label_predictions("./data/test.input.txt","./data/assign3_FFNN_lr001.txt", test_predictions) 
