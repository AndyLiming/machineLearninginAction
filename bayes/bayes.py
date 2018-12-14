import numpy as np
import matplotlib as mpl

def loadDataSet():
  postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
  classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
  return postingList, classVec

def createVocabList(dataset):
  vocabSet = set([])
  for document in dataset:
    vocabSet = vocabSet | set(document)
  return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
  returnVec = [0] * len(vocabList)
  for word in inputSet:
    if word in vocabList:
      returnVec[vocabList.index(word)] = 1
    else:
      print("the word is not in the vovabulary!")
  return returnVec