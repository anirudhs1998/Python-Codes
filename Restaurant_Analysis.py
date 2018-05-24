import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords

import re
from nltk.stem.porter import PorterStemmer
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier

dataset = pd.read_csv('1-restaurant-train.csv',delimiter = '\t',quoting = 0,header=None)
dataset = dataset[0:5000]

corpus = []


'''def create_word_features(words):
    my_dict = dict([(word, True) for word in words])
    return my_dict
'''

for i in range(0, 5000):
    review = re.sub('[^a-zA-Z]', ' ', dataset[1][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in (set(stopwords.words('english')) - {'not'})]
    review = ' '.join(review)
    corpus.append(review)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(analyzer='word')
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 0].values
    
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)





'''reviews_1_strings = []
for i in range(0,5000):
    if dataset[0][i] == 1:
        review = corpus[i]
        review = review.split()
        reviews_1_strings.append((create_word_features(review),"1"))
        
        
reviews_2_strings = []
for i in range(0,5000):
    if dataset[0][i] == 2:
        review = corpus[i]
        review = review.split()
        reviews_2_strings.append((create_word_features(review),"2"))
        
reviews_3_strings = []
for i in range(0,5000):
    if dataset[0][i] == 3:
        review = corpus[i]
        review = review.split()
        reviews_3_strings.append((create_word_features(review),'3'))
        
reviews_4_strings = []
for i in range(0,5000):
    if dataset[0][i] == 4:
        review = corpus[i]
        review = review.split()
        reviews_4_strings.append((create_word_features(review),'4'))
        
reviews_5_strings = []
for i in range(0,5000):
    if dataset[0][i] == 5:  
        review = corpus[i]
        review = review.split()
        reviews_5_strings.append((create_word_features(review),'5'))
        
        
train_set = reviews_1_strings[:round(0.75 * reviews_1_strings.__len__())] + reviews_2_strings[:round(0.75 * reviews_2_strings.__len__())] + reviews_3_strings[:round(0.75 * reviews_3_strings.__len__())] + reviews_4_strings[:round(0.75 * reviews_4_strings.__len__())] + reviews_5_strings[:round(0.75 * reviews_5_strings.__len__())] 
test_set =  reviews_1_strings[round(0.75 * reviews_1_strings.__len__()):] + reviews_2_strings[round(0.75 * reviews_2_strings.__len__()):] + reviews_3_strings[round(0.75 * reviews_3_strings.__len__()):] + reviews_4_strings[round(0.75 * reviews_4_strings.__len__()):] + reviews_5_strings[round(0.75 * reviews_5_strings.__len__()):]
'''



classifier = NaiveBayesClassifier.train(train_set)
accuracy = nltk.classify.util.accuracy(classifier, y_pred)
        



