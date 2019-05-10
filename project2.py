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
                result.append(adj[0] + "_" + token.lemma_)
            #else:
            #    result.append(token.lemma_)
    #result.append([(token.lemma_ + "_" + token.head.text) for token in doc if token.pos_ in tags])
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
        #if (nb.predict(transformer.transform(df['word'])))
    good = sorted(good, key=lambda x:x[1])
    bad = sorted(bad, key=lambda x:x[1])
    
    #result = words_df.nlargest(columns="count", n = terms)
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
    

reviews = pd.read_csv("Amazon_Unlocked_Mobile_new_test.csv")

reviews_brands = {}

for brand in unique_brands(reviews['Brand']):
    reviews_brands[brand] = reviews[(reviews['Brand']) == brand]['Reviews']

nlp = spacy.load('en_core_web_sm')

nb = MultinomialNB()
X = reviews[(reviews['Rating'] == 5) | (reviews['Rating'] == 1)]['Reviews']
y = reviews[(reviews['Rating'] == 5) | (reviews['Rating'] == 1)]['Rating']

transformer = CountVectorizer(analyzer=clean_document).fit(X)
nb.fit(transformer.transform(X), y)

for brand in reviews_brands.keys():    
    print(brand + ":")
    print(frequent_words(text_process(reviews_brands[brand])))
    
print("Brand independent:")
print(frequent_words(text_process(reviews['Reviews'])))
