#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:46:41 2019

@author: vhais
"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.classify import NaiveBayesClassifier
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report

def clean_document(document):
    stopwords_eng = stopwords.words('english')
    #for doc in document:
    document_tokenized = word_tokenize(document)
    result = [word.lower() for word in document_tokenized if word not in stopwords_eng and word not in string.punctuation]
    return result

def stem_document(document):
    stemmer = PorterStemmer()
    #for doc in document:
    result = [stemmer.stem(word) for word in document]
    return result

#def lem_document(document, tags=['NOUN', 'ADJ']):
#    result = []

def unique_brands(brand_list):
    return list(set(brand_list))    

def text_process(text):
    result = clean_document(text)
    result = stem_document(result)
    return result
    

reviews = pd.read_csv("Amazon_Unlocked_Mobile_new_test.csv")

reviews_brands = {}

for brand in unique_brands(reviews['Brand']):
    reviews_brands[brand] = reviews[(reviews['Brand']) == brand]['Reviews']

print(reviews_brands['Alcatel'])
"""
X = reviews['Reviews']
y = reviews['Rating']
z = reviews['Brand']

transformer = CountVectorizer(analyzer=text_process).fit(X)
#X = transformer.transform(X)
#reviews_pos = reviews[(reviews['Rating'] > 3)]
#reviews_neg = reviews[(reviews['Rating'] < 3)]
#reviews_neu = reviews[(reviews['Rating'] == 3)]

nb = MultinomialNB()

#pos_test, pos_train, pos_st_test, pos_st_train = train_test_split(reviews_pos['Reviews'], reviews_pos['Rating'], test_size=0.3, random_state=101)
#neg_test, neg_train, neg_st_test, neg_st_train = train_test_split(reviews_neg['Reviews'], reviews_neg['Rating'], test_size=0.3, random_state=101)
#neu_test, neu_train, neu_st_test, neu_st_train = train_test_split(reviews_neu['Reviews'], reviews_neu['Rating'], test_size=0.3, random_state=101)
        
test_X, train_X, test_y, train_y, test_z, train_z = train_test_split(X, y, z, test_size=0.3, random_state=101)
train_X = transformer.transform(train_X)
nb.fit(train_X, train_y)

for i in range(0, len(test_X)):
    sentences = transformer.transform(sent_tokenize(test_X[i]))
    #sentences = sent_tokenize(test_X[i])
    for sentence in sentences:
        if (str(camera) in str(sentence)):
            print("here")
            print(nb.predict(sentence))        
preds = nb.predict(test_X)

print(confusion_matrix(test_y, preds))
print('\n')
print(classification_report(test_y, preds))
"""