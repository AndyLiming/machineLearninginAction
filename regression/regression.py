import numpy as np
import math
import random
import os
import matplotlib.pyplot as plt
import json
from time import sleep
import urllib3

def loadDataset(fileName):
  numFeat = len(open(fileName).readline().split('\t'))
  dataMat = []
  labelMat = []
  fr = open(fileName)
  for line in fr.readlines():
    lineArr = []
    curLine = line.strip().split('\t')
    for i in range(numFeat - 1):
      lineArr.append(float(curLine[i]))
    dataMat.append(lineArr)
    labelMat.append(float(curLine[-1]))
  return dataMat, labelMat

def standRegres(xArr, yArr):
  xMat = np.mat(xArr)
  yMat = np.mat(yArr).T
  xTx = xMat.T * xMat
  if np.linalg.det(xTx) == 0.0:
    print("this matrix is singular, cannot do inverse")
    return
  ws = xTx.I * (xMat.T * yMat)
  return ws

def lwlr(testPoint, xArr, yArr, k=1.0):
  xMat = np.mat(xArr)
  yMat = np.mat(yArr).T
  m = np.shape(xMat)[0]
  weights = np.mat(np.eye((m)))
  for j in range(m):
    diffMat = testPoint - xMat[j,:]
    weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))
  xTx = xMat.T * (weights * xMat)
  if np.linalg.det(xTx) == 0.0:
    print("this matrix is singular, cannot do inverse")
    return
  ws = xTx.I * (xMat.T * (weights*yMat))
  return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k=1.0):
  m = np.shape(testArr)[0]
  yHat = np.zeros(m)
  for i in range(m):
    yHat[i] = lwlr(testArr[i], xArr, yArr, k)
  return yHat

def ridgeRegres(xMat, yMat, lam=0.2):
  xTx = xMat.T * xMat
  denom = xTx + np.eye(np.shape(xMat)[1]) * lam
  if np.linalg.det(denom) == 0.0:
    print("this matrix is singular, cannot do inverse")
    return
  ws = denom.I * (xMat.T * yMat)
  return ws

def ridgeTest(xArr, yArr):
  xMat = np.mat(xArr)
  yMat = np.mat(yArr).T
  yMean = np.mean(yMat, 0)
  yMat = yMat - yMean
  xMeans = np.mean(xMat, 0)
  xVar = np.var(xMat, 0)
  xMat = (xMat - xMeans) / xVar
  numTestPts = 30
  wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
  for i in range(numTestPts):
    ws = ridgeRegres(xMat, yMat, math.exp(i - 10))
    wMat[i,:] = ws.T
  return wMat

def regularize(xMat):
  inMat = xMat.copy()
  inMeans = np.mean(inMat,0)
  inVar = np.var(inMat,0)
  inMat = (inMat - inMeans)/inVar
  return inMat

def stageWise(xArr, yArr, eps=1.0, numIt=100):
  xMat = np.mat(xArr)
  yMat = np.mat(yArr).T
  yMean = np.mean(yMat, 0)
  yMat = yMat - yMean
  xMat = regularize(xMat)
  m, n = np.shape(xMat)
  returnMat = np.zeros((numIt, n))
  ws = np.zeros((n, 1))
  wsTest = ws.copy()
  wsMax = ws.copy()
  for i in range(numIt):
    print(ws.T)
    lowestError = np.inf
    for j in range(n):
      for sign in [-1, 1]:
        wsTest = ws.copy()
        wsTest[j] += eps * sign
        yTest = xMat * wsTest
        rssE = rssError(yMat.A, yTest.A)
        if rssE < lowestError:
          lowestError = rssE
          wsMax = wsTest
    ws = wsMax.copy()
    returnMat[i,:] = ws.T
  return returnMat

def rssError(yArr, yHatArr):
  return ((yArr-yHatArr)**2).sum() #SSD

def searchForset(retX, retY, setNum, yr, numPce, origPrc):
  sleep(10)
  myApiStr = 'AIzaSyD2cR2KFyx12hXu6PFU-wrWot3NXvko8vY'
  searchURL = 'https://www.googleapis.com/shopping/search/v1/public/products?key=%s&country=US&q=lego+%d&alt=json' % (myAPIstr, setNum)
  pg = urllib3.urlopen(searchURL)
  retDict = json.loads(pg.read())
  for i in range(len(retDict['items'])):
    try:
      currItem = retDict['items'][i]
      if currItem['product']['condition'] == 'new':
        newFlag = 1
      else: newFlag = 0
      listOfInv = currItem['product']['inventories']
      for item in listOfInv:
        sellingPrice = item['price']
        if  sellingPrice > origPrc * 0.5:
          print "%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice)
          retX.append([yr, numPce, newFlag, origPrc])
          retY.append(sellingPrice)
    except: print 'problem with item %d' % i

def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)

def crossValidation(xArr, yArr, numVal=10):
  m = len(yArr)
  indexList = list(range(m))
  errorMat = np.zeros((numVal, 30))
  for i in range(numVal):
    trainX = []
    testX = []
    trainY = []
    testY = []
    random.shuffle(indexList)
    for j in range(m):#create training set based on first 90% of values in indexList
      if j < m*0.9: 
        trainX.append(xArr[indexList[j]])
        trainY.append(yArr[indexList[j]])
      else:
        testX.append(xArr[indexList[j]])
        testY.append(yArr[indexList[j]])
      wMat = ridgeTest(trainX,trainY)    #get 30 weight vectors from ridge
      for k in range(30):#loop over all of the ridge estimates
          matTestX = mat(testX); matTrainX=mat(trainX)
          meanTrain = mean(matTrainX,0)
          varTrain = var(matTrainX,0)
          matTestX = (matTestX-meanTrain)/varTrain #regularize test with training params
          yEst = matTestX * mat(wMat[k,:]).T + mean(trainY)#test ridge results and store
          errorMat[i,k]=rssError(yEst.T.A,array(testY))
          #print errorMat[i,k]
  meanErrors = mean(errorMat,0)#calc avg performance of the different ridge weight vectors
  minMean = float(min(meanErrors))
  bestWeights = wMat[nonzero(meanErrors==minMean)]
  xMat = mat(xArr); yMat=mat(yArr).T
  meanX = mean(xMat,0); varX = var(xMat,0)
  unReg = bestWeights/varX
  print "the best model from Ridge Regression is:\n",unReg
  print "with constant term: ",-1*sum(multiply(meanX,unReg)) + mean(yMat)

if __name__ == '__main__':
  # xArr, yArr = loadDataset('ex0.txt')
  # ws = standRegres(xArr, yArr)
  # print(ws)
  # xMat = np.mat(xArr)
  # yMat = np.mat(yArr)
  # fig = plt.figure()
  # ax = fig.add_subplot(111)
  # ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
  # xCopy = xMat.copy()
  # xCopy.sort(0)
  # yHat = xCopy * ws
  # ax.plot(xCopy[:, 1], yHat)
  # plt.show()

  # yHat = lwlrTest(xArr, xArr, yArr, 0.03)
  # xMat = np.mat(xArr)
  # yMat = np.mat(yArr)
  # srtInd = xMat[:, 1].argsort(0)
  # xSort = xMat[srtInd][:, 0,:]
  # fig = plt.figure()
  # ax = fig.add_subplot(111)
  # ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0],s=2,c='red')
  # ax.plot(xSort[:, 1], yHat[srtInd])
  # plt.show()

  xArr, yArr = loadDataset('abalone.txt')
  fig = plt.figure()
  ax = fig.add_subplot(111)
  # ridgeWeights = ridgeTest(abX, abY)
  # fig = plt.figure()
  # ax = fig.add_subplot(111)
  # ax.plot(ridgeWeights)
  # plt.show()

  # least square #
  xMat = np.mat(xArr)
  yMat = np.mat(yArr).T
  yMean = np.mean(yMat, 0)
  yMat = yMat - yMean
  xMat = regularize(xMat)
  weightsLs = standRegres(xMat, yMat.T).T
  #-------------------#

  weightsSw = stageWise(xArr, yArr, 0.01, 200)
  
  print("LS: ", weightsLs)
  print("stageWise: ", weightsSw[-1])
  ax.plot(weightsSw)
  plt.show()