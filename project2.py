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
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

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
                result.append(adj[0] + "_" + token.lemma_)
    return result

def format_list(li, terms):
    result = ""
    for i in range(0, len(li)):
        if (i < terms):
            result += str(li[i][0])
            result += "\t"
            result += li[i][1]
            result += "\n"
        else:
            break
    return result

def format_result(good, bad, terms):
    result = "possitive:\n"
    result += format_list(good, terms/2)
    result += "negative:\n"
    result += format_list(bad, terms/2)
    return result

def frequent_words(x, terms = 30):
    all_words = ' '.join([text for text in x])
    all_words = all_words.split()
    freq_dist = FreqDist(all_words)
    x = transformer.transform(word.replace("_", " ") for word in freq_dist.keys())
    words_df = pd.DataFrame({'word': list(freq_dist.keys()), 'count': list(freq_dist.values()), 'vector': list(x)})
    good = []
    bad = []
    for i in range(1, len(words_df)):
        if (nb.predict(words_df.at[i, 'vector']) == 5):
            good.append([words_df.at[i, 'count'], words_df.at[i, 'word'].replace(" ", "_")])
        else:
            bad.append([words_df.at[i, 'count'], words_df.at[i, 'word'].replace(" ", "_")])
    good = sorted(good, key=lambda x:x[0], reverse=True)
    bad = sorted(bad, key=lambda x:x[0], reverse=True)
    return format_result(good, bad, terms)

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
    
# pandas load the csv file 
print("Loading data...")
reviews = pd.read_csv("Amazon_Unlocked_Mobile_new_test.csv")
print("Loaded...")

reviews_brands = {}

for brand in unique_brands(reviews['Brand']):
    reviews_brands[brand] = reviews[(reviews['Brand']) == brand]['Reviews']

nlp = spacy.load('en_core_web_sm')

nb = MultinomialNB()
X = reviews[(reviews['Rating'] == 5) | (reviews['Rating'] == 1)]['Reviews']
y = reviews[(reviews['Rating'] == 5) | (reviews['Rating'] == 1)]['Rating']

transformer = CountVectorizer(analyzer=clean_document).fit(X)

test_X, train_X, test_y, train_y = train_test_split(X, y, test_size=0.3, random_state=101)
train_X = transformer.transform(train_X)
nb.fit(train_X, train_y)
test_X = transformer.transform(test_X)
preds = nb.predict(test_X)

print(confusion_matrix(test_y, preds))
print('\n')
print(classification_report(test_y, preds))

print("Learning...")
nb.fit(transformer.transform(X), y)
print("Done learning...")

for brand in reviews_brands.keys():    
    print(brand + ":")
    print(frequent_words(text_process(reviews_brands[brand])))
    
print("Brand independent:")
print(frequent_words(text_process(reviews['Reviews'])))
