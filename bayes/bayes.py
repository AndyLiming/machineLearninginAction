import numpy as np
import matplotlib as mpl
import math
import re
import random

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

def trainNB0(trainMat, trainCate):
  numTrainDocs = len(trainMat)
  numWords = len(trainMat[0])
  pAbusive = sum(trainCate) / float(numTrainDocs)
  p0Num = np.zeros(numWords)
  p1Num = np.zeros(numWords)
  p0Denom = 2.0
  p1Denom = 2.0 #correction
  for i in range(numTrainDocs):
    if trainCate[i] == 1:
      p1Num += trainMat[i]
      p1Denom += sum(trainMat[i])
    else:
      p0Num += trainMat[i]
      p0Denom += sum(trainMat[i])
  p1Vect = p1Num / p1Denom
  p0Vect = p0Num / p0Denom
  return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
  p1 = sum(vec2Classify * p1Vec) + math.log(pClass1)
  p0 = sum(vec2Classify * p0Vec) + math.log(1.0- pClass1)
  return 1 if p1>p0 else 0

def testingNB():
  listOPost, listClasses = loadDataSet()
  myVocabList = createVocabList(listOPost)
  trainMat = []
  for postinDoc in listOPost:
    trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
  p0V, p1V, pAb = trainNB0(trainMat, listClasses)
  testEntry = ['love', 'my', 'dalmation']
  thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
  print(testEntry, "classified as: ", classifyNB(thisDoc, p0V, p1V, pAb))
  testEntry = ['stupid', 'garbage']
  thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
  print(testEntry,"classified as: ",classifyNB(thisDoc,p0V, p1V, pAb))

def bagOfWords2vecMN(vocabList, inputSet):
  retVec = [0] * len(vocabList)
  for word in inputSet:
    if word in vocabList:
      retVec[vocabList.index(word)] += 1
  return retVec

def textParse(bigString):
  listOfTokens = re.split(r'\W*', bigString)
  return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
  docList = []
  classList = []
  fullText = []
  for i in range(1, 26):
    wordList = textParse(open('email/spam/%d.txt' % i).read())
    docList.append(wordList)
    fullText.append(wordList)
    classList.append(1)
    wordList = textParse(open('email/ham/%d.txt' % i).read())
    docList.append(wordList)
    fullText.append(wordList)
    classList.append(0)
  vocabList = createVocabList(docList)
  trainingSet = list(range(50))
  testSet = []
  for i in range(10):
    randIndex = int(random.uniform(0, len(trainingSet)))
    testSet.append(trainingSet[randIndex])
    del (trainingSet[randIndex])
  trainMat = []
  trainClasses = []
  for docIndex in trainingSet:
    trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
    trainClasses.append(classList[docIndex])
  p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
  errorCnt = 0
  for docIndex in testSet:
    wordVector = setOfWords2Vec(vocabList, docList[docIndex])
    if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
      errorCnt += 1
  print("the error rate is %f"%(float(errorCnt)/len(testSet)))

if __name__ == '__main__':
  spamTest()