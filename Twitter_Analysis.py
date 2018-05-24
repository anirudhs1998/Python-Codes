# -*- coding: utf-8 -*-
"""
Created on Tue May 22 10:36:56 2018

@author: Anirudh S
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
import re
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('twitter_samples')
from nltk.corpus import twitter_samples

def create_word_features(words):
    
    words = [ re.sub('[^a-zA-Z@]', ' ', word) for word in words]
    words = [word for word in words if not '@' in word]
    words = [word.lower() for word in words]
    useful_words = [word for word in words if word not in (set(stopwords.words("english"))-{'not'})]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict

neg_strings = twitter_samples.strings('negative_tweets.json')
pos_strings = twitter_samples.strings('positive_tweets.json')

neg_reviews = []
for i in range(0,5000):
    tweet = neg_strings[i]
    tweet = tweet.lower()
    tweet = tweet.split()
    neg_reviews.append((create_word_features(tweet), "negative"))
    
pos_reviews = []
for i in range(0,5000):
    tweet = pos_strings[i]
    tweet = tweet.lower()
    tweet = tweet.split()
    pos_reviews.append((create_word_features(tweet), "positive"))

train_set = neg_reviews[:4000] + pos_reviews[:4000]
test_set =  neg_reviews[4000:] + pos_reviews[4000:]

classifier = NaiveBayesClassifier.train(train_set)
tweet_test = 'Weather '
tweet_test = tweet_test.split()
tweet_test = create_word_features(tweet_test)
classifier.classify(tweet_test)
accuracy = nltk.classify.util.accuracy(classifier, test_set)

