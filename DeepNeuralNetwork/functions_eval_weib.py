import _pickle
import os
import os.path
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import random
import numpy 
import numpy.random as random
import tensorflow as tf
import time
import math
import sys
from keras import backend as K




# Dicts take the string of a list of parameters as hash and returns a list of runtimes (as strings)

glob_weibDict = {}
glob_logDict = {}
glob_parDict = {}




# ~ 20% is for Training, ~ 80% is for testing (probabilistic) (see Pareto principle)
def createRuntimeData():
    trainingSetRatio = 1
    
    #weibtraining = []
    #weibtrainingLabels = []
    weibtraining = []
    weibtrainingLabels = []
    weibDict = {}
    
    weibtesting = []
    weibtestingLabels = []
    
    locs = []
    
    
    trainingInstances = []
    testing = []
    testingLabels = []
    testingInstances = []
    featPath = './chosen_data/chosen_feat/'
    distPath = './chosen_data/chosenDists/'
    files = os.listdir(featPath)
    
    for feat in files:
        if not os.path.isfile(featPath+feat):
            continue
        dist = feat.replace('cnf.feat','dist')

        if os.path.isfile(distPath + dist):      
                
            # open distfile to get the best fitted distribution
            with open(distPath + dist, 'r') as f:
                lines = f.readlines()

                if float(lines[9].split()[1]) < float(lines[17].split()[1]):
                    continue 
                
                locs.append(int(lines[1].split()[1]))
                features = None                        
                # open the feature file to get the feature vector
                with open(featPath + feat, 'r') as fl:
                    fl.readline()
                    line = fl.readline()
                    features = list(map(float,line.split(','))) 
                    features = [features[i] for i in [ 0, 1, 2, 3, 13, 15, 16, 22, 24, 25, 32, 34, 38, 40, 41, 43, 46, 53, 54, 55, 56, 57, 58, 59, 60, 61, 66, 68, 69, 70, 71, 72, 73, 78]] # from variance reduction
                    # features = [features[i] for i in [ 0, 1, 2, 4, 5, 7, 8, 9,10,11,12,13,14,15,20,21,22,23,24,25,26,32,33,34,35]] # from lassoCV Feature Selection
                # open the output file to get the runtimes
                
                with open(featPath.replace('feat/','out/') + feat.replace('.feat','.result')) as fl:
                    runtimes = fl.readlines()
                a= []
                for run in runtimes:
                    a.append(int(run))
                a.sort()
                
                X= []
                Y= []
                weibtraining.append(features)
                weibtrainingLabels.append(a)
                # for run in a:
                #     # weibtraining.append(features)
                #     # weibtrainingLabels.append([run,i/l,shape,scale,metric])
                #     weibtraining.append((a,features))
                #     weibtrainingLabels.append([run-loc,i/l,shape,scale])
                #     i = i+1               
                            
                                
    return weibtraining , weibtrainingLabels, locs

def createRuntimeData_logn():
    trainingSetRatio = 1
    
    #weibtraining = []
    #weibtrainingLabels = []
    weibtraining = []
    weibtrainingLabels = []
    weibDict = {}
    
    weibtesting = []
    weibtestingLabels = []
    
    locs = []
    
    
    trainingInstances = []
    testing = []
    testingLabels = []
    testingInstances = []
    featPath = './chosen_data/chosen_feat/'
    distPath = './chosen_data/chosenDists/'
    files = os.listdir(featPath)
    
    for feat in files:
        dist = feat.replace('cnf.feat','dist')
        if os.path.isfile(distPath + dist):
            
                
            # open distfile to get the best fitted distribution
            with open(distPath + dist, 'r') as f:
                lines = f.readlines()

                if float(lines[9].split()[1]) >= float(lines[17].split()[1]):
                    continue 
                
                locs.append(int(lines[1].split()[1]))
                features = None                        
                # open the feature file to get the feature vector
                with open(featPath + feat, 'r') as fl:
                    fl.readline()
                    line = fl.readline()
                    features = list(map(float,line.split(','))) 
                    features = [features[i] for i in [ 0, 1, 2, 3, 13, 15, 16, 22, 24, 25, 32, 34, 38, 40, 41, 43, 46, 53, 54, 55, 56, 57, 58, 59, 60, 61, 66, 68, 69, 70, 71, 72, 73, 78]] # from variance reduction
                    # features = [features[i] for i in [ 0, 1, 2, 4, 5, 7, 8, 9,10,11,12,13,14,15,20,21,22,23,24,25,26,32,33,34,35]] # from lassoCV Feature Selection
                # open the output file to get the runtimes
                
                with open(featPath.replace('feat/','out/') + feat.replace('.feat','.result')) as fl:
                    runtimes = fl.readlines()
                a= []
                for run in runtimes:
                    a.append(int(run))
                a.sort()
                
                X= []
                Y= []
                weibtraining.append(features)
                weibtrainingLabels.append(a)
                # for run in a:
                #     # weibtraining.append(features)
                #     # weibtrainingLabels.append([run,i/l,shape,scale,metric])
                #     weibtraining.append((a,features))
                #     weibtrainingLabels.append([run-loc,i/l,shape,scale])
                #     i = i+1               
                            
                                
    return weibtraining , weibtrainingLabels, locs


def normalize(data, mean = [], var = []):

    columns = len(data[0])
    normalizedData = []
    if len(mean) == 0:
        mean = numpy.mean(data,axis=0)

    normalizedData = [[x1-x2 for (x1,x2) in zip(x3,mean)] for x3 in data]
    
    #maxi = numpy.absolute(numpy.amax(data, axis=0))
    #mini = numpy.absolute(numpy.amin(data,axis=0))
    #div = numpy.maximum(maxi,mini)
    
    if len(var) == 0:
        var = numpy.var(data,axis=0)

    normalizedData = [[x1/numpy.sqrt(x2) if x2 != 0 else 0 for (x1,x2) in zip(x3,var)] for x3 in normalizedData]
    #print(normalizedData)
    #print("\ntest normalize :",len(data),"\n")
    return normalizedData, mean, var

def sample(data, k=0.1, seed=1909, dist_names=False):
    random.seed(seed)
    random.shuffle(data)
    resultX = []
    resultY=[]
    resultDists = []
    p = 1
    n = len(data)
    while p*k <= 1: 
        if dist_names:
            resultDists.append([x[0] for x in data[int((p-1)*k*n):int(p*k*n)]])
        X = [x[1] for x in data[int((p-1)*k*n):int(p*k*n)]]
        Y = [x[2] for x in data[int((p-1)*k*n):int(p*k*n)]]
        resultX.append(X)
        resultY.append(Y)
        p += 1

    if dist_names:
        return resultDists, resultX, resultY
    return resultX, resultY
    
