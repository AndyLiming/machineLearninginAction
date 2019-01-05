import numpy as np
import math
import matplotlib.pyplot as plt
import random

def loadDataset(filename):
  dataMat = []
  labelMat = []
  with open(filename) as fr:
    for line in fr.readlines():
      lineArr = line.strip().split()
      dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
      labelMat.append(int(lineArr[2]))
  return dataMat, labelMat

def sigmoid0(inX):
  return 1.0 / (1 + np.exp(-inX))

def sigmoid(inX):
  if inX >= 0:
    return 1.0 / (1 + np.exp(-inX))
  else:
    return np.exp(inX) / (1 + np.exp(inX))

def gradAscent(dataMatIn, labelMatIn):
  dataMatrix = np.mat(dataMatIn)
  labelMatrix = np.mat(labelMatIn).transpose()
  m, n = np.shape(dataMatrix)
  print(m,n)
  alpha = 0.001  #learning rate
  maxCycle = 500  #iteration times
  weights = np.ones((n, 1))  #init weights
  for k in range(maxCycle):
    h = sigmoid0(dataMatrix * weights)
    error = labelMatrix - h
    weights = weights + alpha * dataMatrix.transpose() * error
  weights=np.mat(weights).reshape((3,1))
  return weights

def stocGradAscent0(dataMatrix, classLabels,numIter=150):
  m, n = np.shape(dataMatrix)
  dataMatrix = np.array(dataMatrix)
  weights = np.ones(n)
  for k in range(numIter):
    dataIndex=list(range(m))
    for i in range(m):
      alpha = 4 / (1.0+ i + k) + 0.01  #learning rate
      randIndex=int(random.uniform(0,len(dataIndex)))
      h = sigmoid(sum(dataMatrix[randIndex] * weights))
      error = classLabels[randIndex] - h
      weights = weights + alpha * error * dataMatrix[randIndex]
      del(dataIndex[randIndex])
  return weights

def plotBestFit(weights, dataMat, labelMat):
  dataArr = np.array(dataMat)
  n = np.shape(dataArr)[0]
  xcord1 = []
  xcord2 = []
  ycord1 = []
  ycord2 = []
  for i in range(n):
    if int(labelMat[i]) == 1:
      xcord1.append(dataArr[i, 1])
      ycord1.append(dataArr[i, 2])
    else:
      xcord2.append(dataArr[i, 1])
      ycord2.append(dataArr[i, 2])
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
  ax.scatter(xcord2, ycord2, s=30, c='green')
  x = np.arange(-3.0, 3.0, 0.1)
  y = ((-weights[0] - weights[1] * x) / weights[2]).transpose()#transpose to keep dimension ocnsistancy
  print(x.shape,y.shape)
  ax.plot(x, y)
  plt.xlabel('X1')
  plt.ylabel('X2')
  plt.show()

def classifyVector(inX, weights):
  prob = sigmoid(sum(inX * weights))
  if prob > 0.5: return 1.0
  else: return 0.0

def colicTest():
  frTrain = open('horseColicTraining.txt')
  frTest = open('horseColicTest.txt')
  trainingSet = []
  trainingLabels = []
  for line in frTrain.readlines():
    currLine = line.strip().split('\t')
    lineArr = []
    for i in range(21):
      lineArr.append(float(currLine[i]))
    trainingSet.append(lineArr)
    trainingLabels.append(float(currLine[21]))
  trainWeights = stocGradAscent0(trainingSet, trainingLabels, 500)
  errorCount = 0
  numTestVec = 0
  for line in frTest.readlines():
    numTestVec += 1.0
    currLine = line.strip().split('\t')
    lineArr = []
    for i in range(21):
      lineArr.append(float(currLine[i]))
    if int(classifyVector(np.array(lineArr), trainWeights)) != int(currLine[21]):
      errorCount += 1
  errorRate = (float(errorCount) / numTestVec)
  print("the error rate of this : %f" % errorRate)
  return errorRate

def multiTest():
  numTests = 10
  errorSum = 0.0
  for k in range(numTests):
    errorSum += colicTest()
  print("after %d iterations the average error rate is: %f"%(numTests,errorSum/float(numTests)))

if __name__ == '__main__':
  dataMat, labelMat = loadDataset('testSet.txt')
  #weights = gradAscent(dataMat, labelMat)
  #weights = stocGradAscent0(dataMat, labelMat,50)
  #print(weights)
  #plotBestFit(weights, dataMat, labelMat)
  multiTest()