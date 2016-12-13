#!/usr/bin/python
#-*-coding:utf-8-*-
'''@author:duncan'''

import pandas as pd
import svmMLiA as svm
from sklearn import cross_validation
from numpy import *

def loadDataSet(filename):
    df = pd.read_excel(filename)
    testdata = df.drop(['survived'],axis=1).values
    # we need to convert value 0 of survived into -1 to satisfy
    labeldata = df['survived'].values
    for i in range(len(labeldata)):
        if labeldata[i] == 0:
            labeldata[i] = -1
    # 2-8 to split the real data
    trainingdataarr,testdataarr,traininglabelarr,testlabelarr = cross_validation.train_test_split(testdata,labeldata,test_size=0.2)
    return trainingdataarr,traininglabelarr,testdataarr,testlabelarr

def titanicBysvm(filename,k1=1.3):
    trainingdataarr,traininglabelarr,testdataarr,testlaeblarr = loadDataSet(filename)
    # C = 200 , toler = 0.0001 , maxiteration = 10000
    # b,alphas = svm.smoSimple(trainingdataarr,traininglabelarr,100,0.001,10000)
    b,alphas = svm.smoP(trainingdataarr,traininglabelarr, 200, 0.0001, 10000, ('rbf', k1))
    # w f(x) = w.T*x + b
    w = svm.calcWs(alphas,trainingdataarr,traininglabelarr)
    dataMat = mat(trainingdataarr); labelmat = mat(traininglabelarr).transpose()
    # svInd is the index of support vector,sVs is the support vectors,labelSV is the support vector labels
    svInd = nonzero(alphas.A > 0)[0]
    sVs = dataMat[svInd]
    labelSV = labelmat[svInd]
    print "there are %d Support Vectors" % shape(sVs)[0]
    m,n = shape(trainingdataarr)
    errorCount = 0
    for i in range(m):
        kernelEval = svm.kernelTrans(sVs,dataMat[i,:], ('rbf',k1))
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        # predict = trainingdataarr[i,:] * mat(w) + b
        # print predict
        # print trainingdataarr[i,:]
        # print "  "
        # print predict
        # print "\n"
        if sign(predict) != sign(traininglabelarr[i]):
            errorCount += 1
    print "the number of errors is %d" % errorCount
    print "the training accuracy rate is: %f"  % (1 - (float)(errorCount) / m)

    # test data running
    testErrorCount = 0
    dataMat = mat(testdataarr); testlabelMat = mat(testlaeblarr).transpose()
    m,n = shape(testdataarr)
    for i in range(m):
        kernelEval = svm.kernelTrans(sVs,dataMat[i,:],('rbf',k1))
        predict = kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict) != sign(testlaeblarr[i]):
            testErrorCount += 1
    print "the test accuracy rate is %.3f" % (1 - float(testErrorCount) / m)

if __name__ == "__main__":
    titanicBysvm('../data.xls')