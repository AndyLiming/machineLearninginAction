import numpy as np
import math

def loadDataset(filename):
  dataMat = []
  labelMat = []
  with open(filename) as fr:
    for line in fr.readlines():
      lineArr = line.strip().split()
      dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
      labelMat.append(int(lineArr[2]))
  return dataMat, labelMat

def sigmoid(inX):
  return 1.0 / (1 + np.exp(-inX))

def gradAscent(dataMatIn, labelMatIn):
  dataMatrix = np.mat(dataMatIn)
  labelMatrix = np.mat(labelMatIn).transpose()
  m, n = np.shape(dataMatrix)
  print(m,n)
  alpha = 0.001  #learning rate
  maxCycle = 500  #iteration times
  weights = np.ones((n, 1))  #init weights
  for k in range(maxCycle):
    h = sigmoid(dataMatrix * weights)
    error = labelMatrix - h
    weights = weights + alpha * dataMatrix.transpose() * error
  return weights

if __name__ == '__main__':
  dataMat, labelMat = loadDataset('testSet.txt')
  weights = gradAscent(dataMat, labelMat)
  print(weights)