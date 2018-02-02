'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
                [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

def calcShannonEnt(dataSet):
    """
    计算香农熵
    """
    # 计算有多少个样本
    numEntries = len(dataSet)
    labelCounts = {}
    # 遍历每个样本
    for featVec in dataSet: #the the number of unique elements and their occurance
        # 得到这个样本的标签
        currentLabel = featVec[-1]
        # 标签不在子集中
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt
    
def splitDataSet(dataSet, axis, value):
    """
    划分数据集
    dataSet: 要划分的数据集
    axis: 依据的特征
    """
    retDataSet = []
    # 遍历每个样本
    for featVec in dataSet:
        if featVec[axis] == value:
            # 去掉划分数据用的 axis 特征列
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
    
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    # 提取所有样例的标签
    classList = [example[-1] for example in dataSet]
    # 递归结束条件1：当前结点的所有样例的标签都相同
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    # 递归结束条件2：所有的特征都用完了
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    # 选出最好的特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    # 创建根结点
    myTree = {bestFeatLabel:{}}
    # 删掉已用的特征
    del(labels[bestFeat])
    # 得到对应的特征的所有取值的集合
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    # 遍历对应特征的所有取值
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        # 递归的构造子树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            
    
def classify(inputTree,featLabels,testVec):
    # 树的根结点
    firstStr = inputTree.keys()[0]
    # 该根结点的子树
    secondDict = inputTree[firstStr]
    # 根节点特征对应的位置
    featIndex = featLabels.index(firstStr)
    # 传入的待测试样例的该位置的特征值
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    # 判断到不到叶子结点
    if isinstance(valueOfFeat, dict):
        # 不是叶子结点，递归 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    # 到叶子结点了，那么就代表着分类结束了
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
