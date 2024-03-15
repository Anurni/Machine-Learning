#! /usr/bin/env python3
# -*- encoding: utf-8 -*-

"""
Anni Nieminen, assignment 2

This program trains and evaluates (with devset) a multiclass linear regression model,
which classifies tweets into three categories: positive, neutral, and negative.

"""

# declaring imports

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
import sklearn.metrics
import pandas as pd

# reading the data and loading it to memory

def load_data(filepath):

    """Reads the data file with Panda's read_cvs-method.
   Saves tweets(features) from the file in a list.
    Saves labels(classes) in a list.
    Returns:
        list that contains the tweets
        list that contains the labels
        label imbalance (only for later model evaluation purposes)"""
    
    data = pd.read_csv(filepath, sep="\t")

    only_tweets = []
    only_labels = []
    
    for index, row in data.iterrows():
        only_tweets.append(row.iloc[2])
        only_labels.append(row.iloc[1])

   #    checking label imbalance for evaluation metrics later
    pos = 0
    neg = 0 
    neut = 0  
    for label in only_labels:       
        if label == "positive":
            pos += 1
        elif label == "negative":
            neg += 1
        else:
            neut += 1
    label_imbalance = (f"num of positive labels: {pos}, num of negative labels: {neg}, num of neutral labels: {neut}")

    return only_tweets, only_labels, label_imbalance

#   Getting training, development, and testing(input) data

training_data = load_data("C:\\Users\\annin\\assignment2\\data\\training.txt")
development_data = load_data("C:\\Users\\annin\\assignment2\\data\\development.gold.txt")
testing_data_input = load_data("C:\\Users\\annin\\assignment2\\data\\test.input.txt")

#   Vectorizing our tweets (features) and encoding our three classes with LabelEncoder
#   Implementing the BOW-method with CountVectorizer

training_tweets, training_labels, training_label_imbalance = training_data
vectorizer = CountVectorizer(stop_words='english')      #   ended up using only stopwords as the other parameters didn't improve model's performance
X_train = vectorizer.fit_transform(training_tweets)

# encoding our labels with LabelEncoder

lb = preprocessing.LabelEncoder() 
Y_Train = lb.fit_transform(training_labels) 

#   Building and training our Logistic Regression model, modified number of iterations

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_Train) #   Fitting our model to the training data


#   Transforming our dev data and dev labels

development_tweets, development_labels, development_label_imbalance = development_data
X_dev = vectorizer.transform(development_tweets)
Y_dev = lb.transform(development_labels)

#   Getting the predictions

y_pred = model.predict(X_dev)

#   Working on test.input.txt data, vectorizing the features and predicting the labels

test_tweets, test_labels, test_label_imbalance = testing_data_input  #  here, we're only going to need the tweets
test_tweets_vectorized = vectorizer.transform(test_tweets)
test_tweet_predictions = model.predict(test_tweets_vectorized)
test_tweet_predictions = lb.inverse_transform(test_tweet_predictions) #   transforming the labels back to human-readable form
print(len(test_tweet_predictions))

#   Writing our model's label predictions to development.input.txt and test.input.txt
#   Creating a function to ease this

def write_label_predictions(filepath, new_filepath, pred_labels):
    data = pd.read_csv(filepath, sep="\t")
    data.iloc[:,1] = pred_labels
    data.to_csv(new_filepath, sep="\t", index=False)
  
#Now, we can write the label predictions to new txt files
#development.input.txt --> development.input.predicted.labels.txt
#Here we need to decode the label back into a human-readable form "positive", "negative" and "neutral"
write_label_predictions("C:\\Users\\annin\\assignment2\\data\\development.input.txt", 
                        "C:\\Users\\annin\\assignment2\\data\\development.input.predicted.labels.txt", lb.inverse_transform(y_pred))

#test.input.txt --> assign2_logReg_L2norm.txt
write_label_predictions("C:\\Users\\annin\\assignment2\\data\\test.input.txt", 
                        "C:\\Users\\annin\\assignment2\\data\\assign2_logReg_L2norm.txt", test_tweet_predictions) 

#MODEL EVALUATION STARTS HERE

def evaluate_model():
    print("**** MODEL EVALUATION *****")

    accuracy = sklearn.metrics.accuracy_score(Y_dev, y_pred)
    print(f"The accuracy is {accuracy}.")

    recall = sklearn.metrics.recall_score(Y_dev, y_pred, average="weighted")  # using weighted after having observed the label imbalance, is that the way to go?
    print(f"The recall score is {recall}")

    precision = sklearn.metrics.precision_score(Y_dev, y_pred, average="weighted")
    print(f"The precision score is {precision}")

    f1 = sklearn.metrics.f1_score(Y_dev, y_pred, average="weighted")
    print(f"The f1-score is {f1}.")

    training_data_label_imbalance = (f"The label (im)balance in the training data: {training_label_imbalance}")
    print(training_data_label_imbalance)

    development_data_label_imbalance = (f"The label (im)balance in the development data: {development_label_imbalance}")
    print(development_data_label_imbalance)


if __name__ == "__main__":
    evaluate_model()
