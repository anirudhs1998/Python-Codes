import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('all_tickets.csv')
dataset = dataset.dropna()
dataset_urg = dataset.iloc[:,[0,1,7]].values
#dataset_imp = dataset.iloc[:,[0,1,8]].values

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

for i in range(0,1000):
    review = dataset_urg[i][0] + ' ' + dataset_urg[i][1]
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in (set(stopwords.words('english')) - {'not'})]
    review = ' '.join(review)
    corpus.append(review)
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(analyzer = 'word',max_features=4000)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[0:10000, 7].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
    

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

