import numpy as np
import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from nltk.stem.snowball import SnowballStemmer

column_to_predict = "urgency"


class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        stemmer = SnowballStemmer("english", ignore_stopwords=True)
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


stop_words_lang = 'english'
fit_prior = True
min_data_per_class = 1 

#dataframe = pd.read_csv('all_tickets.csv',dtype=str)
dataframe = pd.read_csv('all_tickets.csv')
text_columns = "body"
#bytag = dataframe.groupby(column_to_predict).aggregate(np.count_nonzero)
#tags = bytag[bytag.body >= min_data_per_class].index
#dataframe = dataframe[dataframe[column_to_predict].isin(tags)]

y = dataframe[column_to_predict]
X = dataframe[text_columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#Removing stop words and performing stemming using PIPELINE

count_vect = CountVectorizer(stop_words=stop_words_lang)
#count_vect = StemmedCountVectorizer(stop_words=stop_words_lang)
    
#Using Naive Bayes Classifier
text_clf = Pipeline([('vect', count_vect),('tfidf', TfidfTransformer()),('clf', MultinomialNB(fit_prior=fit_prior))])
#Using SVM Classifier
text_clf = Pipeline([('vect', count_vect),('tfidf', TfidfTransformer()),('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,n_iter=5, random_state=42))])

text_clf = text_clf.fit(X_train, y_train)

'''
#ticket = "Hi, my outlook app seems to misbehave a lot lately. I cannot sync my emails and it often crashes and asks for credentials.Could you help me out"
ticket = input("Enter ticket")
ticket_test = pd.Series(ticket)
ticket_analysis = text_clf.predict(ticket_test)
'''

y_pred = text_clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

prediction_acc = np.mean(y_pred == y_test) 

