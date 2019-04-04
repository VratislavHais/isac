#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:59:39 2019

@author: vhais
"""
from argparse import ArgumentParser
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
#import nltk


"""
def sharedWords(input, output, query):
    for line in input:
        tokens = nltk.word_tokenize(line)
        vectorize(line)
        break
"""
def fileToArray(text):
    result = []
    for line in text:        
        result.append(line)
    return result

parser = ArgumentParser()
parser.add_argument('-f', '--file', default='input.txt', help="File containing documents.")
parser.add_argument('-o', '--out', default='output.txt', help="File containing results.")
parser.add_argument('query', type=str, help="Query.")
args = parser.parse_args()

input1 = open(args.file, encoding="utf-8", errors="ignore")
output = open(args.out,encoding="utf-8", errors="ignore")
vectorizer = CountVectorizer()

array = fileToArray(input1)
array.append(str(args.query))
X = vectorizer.fit_transform(array)


#print(vectorizer.get_feature_names())
input1.close()
output.close()
