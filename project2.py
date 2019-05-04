#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:46:41 2019

@author: vhais
"""

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pandas as pd
from nltk import FreqDist
import spacy

def twograms(nouns_adj):
    for i in range(0, len(nouns_adj)):
        print(nouns_adj[i].pos)
        if (nouns_adj[i].pos == "NOUN" and i > 0 and nouns_adj[i].pos == "ADJ"):
            print(nouns_adj[i])
            print(nouns_adj[i-1])
            nouns_adj[i] = nouns_adj[i-1] + "_" + nouns_adj[i]
            print(nouns_adj[i])
            nouns_adj.pop(i-1)
            print(nouns_adj[i-1])
    return nouns_adj

def clean_document(document):
    stopwords_eng = stopwords.words('english')
    document_tokenized = word_tokenize(document)
    result = [word.lower() for word in document_tokenized if word not in stopwords_eng and word not in string.punctuation]
    result = [word.replace("[^a-zA-Z#]", " ") for word in result]
    result = [word for word in result if len(word) > 2]
    return result

def lem_document(document, tags=['NOUN']):
    result = []
    doc = nlp(" ".join(document))
    
    for token in doc:
        if token.pos_ in tags:
            adj = []
            for adjective in list(token.children):
                if adjective.pos_ == "ADJ":
                    adj.append(adjective.lemma_)
            if (len(adj) > 0):
                result.append(token.lemma_ + "_" + adj[0])
            #else:
            #    result.append(token.lemma_)
    #result.append([(token.lemma_ + "_" + token.head.text) for token in doc if token.pos_ in tags])
    return result

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
        try:
            tmp = clean_document(text)
            result = result + lem_document(tmp)
        except KeyboardInterrupt:
            exit(1)
        except:
            continue
    return result
    

reviews = pd.read_csv("Amazon_Unlocked_Mobile_new_test.csv")

reviews_brands = {}

for brand in unique_brands(reviews['Brand']):

    reviews_brands[brand] = reviews[(reviews['Brand']) == brand]['Reviews']
nlp = spacy.load('en_core_web_sm')

for brand in reviews_brands.keys():    
    print(brand + ":")
    print(frequent_words(text_process(reviews_brands[brand])))
    
print("Brand independent:")
print(frequent_words(text_process(reviews['Reviews'])))
