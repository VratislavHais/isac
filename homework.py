#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:59:39 2019

@author: vhais
"""
from argparse import ArgumentParser

def getLowestIndex(array):
    low = float("inf")
    index = 0
    for i in range(0, len(array)):
        if (low < i[0]):
            low = i[0]
            index = i
    return index
    

def sharedWords(input, output, query):
    result = []
    for line in input:
        if (len(result) < 10):
            result.append([line.count(query), line])
            continue
        occurences = line.count(query)
        if (occurences > result[getLowestIndex(result)][0]):
            result[getLowestIndex(result)] = [occurences, line]
    return result
        

parser = ArgumentParser()
parser.add_argument('-f', '--file', default='input.txt', help="File containing documents.")
parser.add_argument('-o', '--out', default='output.txt', help="File containing results.")
parser.add_argument('query', type=str, help="Query.")
args = parser.parse_args()

file = open(args.file, 'r')
output = open(args.out, 'w')

print(sharedWords(file, output, args.query))

file.close()
output.close()