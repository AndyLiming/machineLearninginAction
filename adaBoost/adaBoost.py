import numpy as np
import math
import random
import os

def loadSimpData():
  dataMat = np.matrix([[1., 2.],
                      [2., 1.1],
                      [1.3, 1.],
                      [1., 1.],
                      [2., 1.]])
  classLabels=[1.0, 1.0, -1.0, -1.0, 1.0]
  return dataMat, classLabels

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
  retArray = np.ones((np.shape(dataMatrix)[0], 1))
  if threshIneq == 'lt':
    retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
  else:
    retArray[dataMatrix[:, dimen] > threshVal] = -1.0
  return retArray

def buildStump(dataArr, classLabels, D):
  dataMatrix = np.mat(dataArr)
  labelMat = np.mat(classLabels).T
  m, n = np.shape(dataMatrix)
  numSteps = 10.0
  bestStump = {}
  bestClassEst = np.mat(np.zeros((m, 1)))
  minErr = np.inf
  for i in range(n):
    rangeMin = dataMatrix[:,i].min()
    rangeMax = dataMatrix[:, i].max()
    stepSize = (rangeMax - rangeMin) / numSteps
    for j in range(-1, int(numSteps) + 1):
      for inequal in ['lt', 'gt']:
        threshVal = (rangeMax + float(j) * stepSize)
        predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
        errArr = np.mat(np.ones((m, 1)))
        errArr[predictedVals == labelMat] = 0
        weightedErr = D.T * errArr
        #print("split: dim %d, thresh %.2f, thersh inequal %s, weighted error %.3f" % (i, threshVal, inequal, weightedErr))
        if weightedErr < minErr:
          minErr = weightedErr
          bestClassEst = predictedVals.copy()
          bestStump['dim'] = i
          bestStump['thersh'] = threshVal
          bestStump['ineq'] = inequal
  return bestStump, minErr, bestClassEst

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
  weakClassArr = []
  m = np.shape(dataArr)[0]
  D = np.mat(np.ones((m, 1)) / m)
  aggClassEst = np.mat(np.zeros((m, 1)))
  for i in range(numIt):
    bestStump, error, classEst = buildStump(dataArr, classLabels, D)
    print(error.dtype)
    print("D: ", D.T)
    alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
    bestStump['alpha'] = alpha
    weakClassArr.append(bestStump)
    print("classEst: ", classEst.T)
    expon=np.multiply(-1 * alpha * np.mat(classLabels).T, classEst)
    D=np.multiply(D, np.exp(expon))
    D=D / D.sum()
    aggClassEst += alpha * classEst
    print("aggClassEst: ", aggClassEst.T)
    aggErrors=np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
    errorRate=aggErrors.sum() / m
    print("total error: ", errorRate)
    if errorRate == 0.0: break
  return weakClassArr

if __name__ == '__main__':
  dataMat, classLabels = loadSimpData()
  #D = np.mat(np.ones((5, 1)) / 5)
  #buildStump(dataMat, classLabels, D)
  adaBoostTrainDS(dataMat, classLabels,9)