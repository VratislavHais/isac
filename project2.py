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
from nltk import FreqDist
import spacy
import re

def clean_document(document):
    stopwords_eng = stopwords.words('english')
    document_tokenized = word_tokenize(document)
    result = [word.lower() for word in document_tokenized if word not in stopwords_eng and word not in string.punctuation]
    result = [word.replace("[^a-zA-Z#]", " ") for word in result]
    result = [word for word in result if len(word) > 2]
    return result

def lem_document(document, tags=['NOUN', 'ADJ']):
    result = []
    #for sent in document:
    doc = nlp(" ".join(document))
    result.append([token.lemma_ for token in doc if token.pos_ in tags])
    return result[0]

def frequent_words(x, terms = 10):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()
    freq_dist = FreqDist(all_words)
    words_df = pd.DataFrame({'word': list(freq_dist.keys()), 'count': list(freq_dist.values())})
    result = words_df.nlargest(columns="count", n = terms)
    return result

def unique_brands(data):
    seen, result = set(), []
    for item in data:
        if item.lower() not in seen:
            seen.add(item.lower())
            result.append(item)
    return result

def text_process(texts):
    result = []
    for text in texts:
        tmp = clean_document(text)
        result = result + lem_document(tmp)
    return result
    

reviews = pd.read_csv("Amazon_Unlocked_Mobile_new_test.csv")

reviews_brands = {}

for brand in unique_brands(reviews['Brand']):

    reviews_brands[brand] = reviews[(reviews['Brand']) == brand]['Reviews']
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

for brand in reviews_brands.keys():    
    print(brand + ":")
    print(frequent_words(text_process(reviews_brands[brand])))
    
print("Brand independent:")
print(frequent_words(text_process(reviews['Reviews'])))
