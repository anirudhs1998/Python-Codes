import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk
import re
nltk.download('movie_reviews')
nltk.download('punkt')

def create_word_features(words):
    words = [ re.sub('[^a-zA-Z]', ' ', word) for word in words]
    words = [word.lower() for word in words]
    useful_words = [word for word in words if word not in (set(stopwords.words("english"))-{'not'})]
    my_dict = dict([(word, True) for word in useful_words])
    return my_dict

'''#########################################333
neg_reviews = []
review_test = 'The Movie was not good. It was so boring'
words = word_tokenize(review_test)
neg_reviews.append((create_word_features(words), "negative"))

############################################'''

neg_reviews = []
for fileid in movie_reviews.fileids('neg'):
    words  = movie_reviews.words(fileid)
    neg_reviews.append((create_word_features(words), "negative"))

pos_reviews = []
for fileid in movie_reviews.fileids('pos'):
    words = movie_reviews.words(fileid)
    pos_reviews.append((create_word_features(words), "positive"))
    
train_set = neg_reviews[:750] + pos_reviews[:750]
test_set =  neg_reviews[750:] + pos_reviews[750:]




classifier = NaiveBayesClassifier.train(train_set)
accuracy = nltk.classify.util.accuracy(classifier, test_set)
review_test = 'good'
words = word_tokenize(review_test)
words = create_word_features(words)
classifier.classify(words)
 
