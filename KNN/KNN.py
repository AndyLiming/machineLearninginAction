import numpy as np
import operator
import matplotlib as mpl
import os

def creatDataset():
  groups = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
  labels = ['A', 'A', 'B', 'B']
  return groups, labels
  
def classify0(inX, dataset, labels, k):
  #inX: 输入向量
  #dataset：训练样本集
  #labels：标签向量
  #k: 最邻近点的数目
  datasetSize = dataset.shape[0]
  diffMat = np.tile(inX, (datasetSize, 1)) - dataset
  sqDiffMat = diffMat ** 2
  sqDistance = sqDiffMat.sum(axis=1)
  distances = sqDistance ** 0.5
  sortedDisId = distances.argsort()
  classCount = {}
  for i in range(k):
    voteLables = labels[sortedDisId[i]]
    classCount[voteLables] = classCount.get(voteLables, 0) + 1
  sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)#reverse=True 降序；reverse=False 升序
  return sortedClassCount[0][0]

def file2matrix(filename):
  f = open(filename)
  arrayOLines = f.readlines()#读取所有行
  numberOfLines = len(arrayOLines)#行数
  resMat = np.zeros((numberOfLines, 3))#3种特征
  classLabelVector = []
  index = 0
  for line in arrayOLines:#逐行读取
    line = line.strip()#去除首尾空格
    listFromLine = line.split('\t')#tab分割
    resMat[index,:] = listFromLine[0:3]#前三列是特征
    classLabelVector.append(int(listFromLine[-1]))#最后一列是类别
    index += 1
  return resMat,classLabelVector

def autoNorm(dataset):
  minVals = dataset.min(0)  #每列取极值
  maxVals = dataset.max(0)
  ranges = maxVals - minVals
  normDataset = np.zeros(np.shape(dataset))
  m = dataset.shape[0]#行数
  normDataset = dataset - np.tile(minVals, (m, 1))
  normDataset = normDataset / (np.tile(ranges, (m, 1)))
  return normDataset,ranges,minVals

def datingClassifyTest(normDatingDataset, datingLabels, testRatio, k):
  m = normDatingDataset.shape[0]
  numTestVecs = int(m * testRatio)
  errorCount = 0.0
  for i in range(numTestVecs):
    classifyRes = classify0(normDatingDataset[i,:], normDatingDataset[numTestVecs:m,:], datingLabels[numTestVecs:m], k)
    print("No = %d,classify result = %d, real label = %d" % (i,classifyRes, datingLabels[i]))
    if (classifyRes != datingLabels[i]):
      errorCount += 1.0
  print("total error rate is %f"%(errorCount/float(numTestVecs)))

def datingClassifyPredict():
  srcPath = 'D:/Project/MLInAction/machinelearninginaction/Ch02/'
  datasetName = 'datingTestSet2.txt'
  fileName = srcPath + datasetName
  dataset, datingLabels = file2matrix(fileName)
  normDatingDataset, ranges, minVals = autoNorm(dataset)
  k=3
  resultList = ['not at all', 'in small doses', 'in large doses']
  precentTats = float(input("percentage of time spent playing video games?"))
  ffMiles = float(input("frequent flier miles earned per year?"))
  icecream = float(input("liters of ice cream consumed per year?"))
  inArr = np.array([ffMiles, precentTats, icecream])
  classsifyRes = classify0((inArr - minVals) / ranges, normDatingDataset, datingLabels, k)
  print("you will probably like this person: %s" % (resultList[classsifyRes - 1]))

def img2vector(filename):
  vec = np.zeros((1, 1024))
  f = open(filename)
  for i in range(32):
    lineStr = f.readline()
    for j in range(32):
      vec[0, 32 * i + j] = int(lineStr[j])
  return vec

def handWrittingClassTest():
  hwLabels = []
  hwSrcPath='D:/Project/MLInAction/machinelearninginaction/Ch02/digits/'
  trainingPath = hwSrcPath + 'trainingDigits'
  testPath = hwSrcPath + 'testDigits'
  k=3
  trainingFileList = os.listdir(trainingPath)
  testFileList = os.listdir(testPath)
  m = len(trainingFileList)
  trainingMat = np.zeros((m, 1024))
  for i in range(m):
    filename = trainingFileList[i]
    fileStr = filename.split('.')[0]
    classNumStr = int(fileStr.split('_')[0])
    hwLabels.append(classNumStr)
    trainingMat[i,:] = img2vector(trainingPath + '/' + filename)
  errorCount=0.0
  mTest=len(testFileList)
  for i in range(mTest):
    filename = testFileList[i]
    fileStr = filename.split('.')[0]
    classNumStr = int(fileStr.split('_')[0])
    vecTest=img2vector(testPath + '/' + filename)
    classifyRes=classify0(vecTest, trainingMat, hwLabels, k)
    print("classified result is %d, real answer is %d" % (classifyRes, classNumStr))
    if (classifyRes != classNumStr):
      errorCount += 1.0
  print("the total number of errors is: %d" % (errorCount))
  print("the error rate is: %f" % (errorCount/float(mTest)))

if __name__ == '__main__':
  #datingClassifyPredict()
  handWrittingClassTest()