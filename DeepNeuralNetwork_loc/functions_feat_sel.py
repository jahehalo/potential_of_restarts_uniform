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
    
    
    
    
    trainingInstances = []
    testing = []
    testingLabels = []
    testingInstances = []
    
    for featPath in [ '../features/cnfs_1500-2500/']: #, ,'../features/n3000/', '../features/cnfs_500-1000/',
        # This path contains the features for all training instances.

        distPath = '../matlab_implementations/ks_test/dists_eval/'
        for r in ['4.14/','4.15/','4.16/','4.17/','4.18/','4.19/','4.20/','4.23/','4.24/','4.25/','4.26/','4.27/']: # 
            
            if ((r in ['4.14/','4.15/','4.16/','4.17/','4.18/','4.19/','4.20/','4.27/']
                and featPath == '../features/cnfs_1500-2500/')
                or (r in ['4.14/','4.15/','4.16/','4.17/','4.18/','4.19/','4.20/','4.27/']
                and featPath == '../features/cnfs_500-1000/')
                or (r in ['4.23/','4.24/','4.25/','4.26/','4.27/']
                and featPath == '../features/n3000/')):
                continue
            
            files = os.listdir(featPath + r)
            for feat in files:
                # This file contains the data for the maximum likelihood estimates.
                dist = feat.replace('cnf.feat','dist')
                
                if os.path.isfile(distPath + dist):
                        
                        
                    # open distfile to get the best fitted distribution
                    with open(distPath + dist, 'r') as f:
                        lines = f.readlines()
                        loc = int(lines[1].split()[1])
                        scale = float(lines[2].split()[1])
                        shape = float(lines[5].split()[1])
                        
                        features = None                        
                        # open the feature file to get the feature vector
                        with open(featPath + r + feat, 'r') as fl:
                            fl.readline()
                            line = fl.readline()
                            features = list(map(float,line.split(','))) 
                            # Only consider the features with a minimum degree of variance.
                            # These features were obtained by first scaling them to a minimum of zero and a maximum of one.
                            # Then, the features with a variance (after scaling) of at least 0.05 were selected.
                            # Four additional features were also added.
                            # The handpicked features are [25, 41, 46, 55].
                            features = [features[i] for i in [ 0, 1, 2, 3, 13, 15, 16, 22, 24, 25, 32, 34, 38, 40, 41, 43, 46, 53, 54, 55, 56, 57, 58, 59, 60, 61, 66, 68, 69, 70, 71, 72, 73, 78]] # from variance reduction
                        
                        X= []
                        Y= []

                        weibtraining.append(features)
                        weibtrainingLabels.append([numpy.log(loc)])            
                                    
                                    
    return weibtraining , weibtrainingLabels 
                                    
def createSampleData():
    """
        This function returns the data for all distributions.
    """

    feats_labels = []


    weibtraining = []
    weibtrainingLabels = []
    weibDict = {}
    
    weibtesting = []
    weibtestingLabels = []

    
    trainingInstances = []
    testing = []
    testingLabels = []
    testingInstances = []
    
    for featPath in [ '../features/cnfs_1500-2500/']: 
        # This path contains the features for all training instances.
    
        distPath = '../matlab_implementations/ks_test/dists/'
        for r in ['4.14/','4.15/','4.16/','4.17/','4.18/','4.19/','4.20/','4.23/','4.24/','4.25/','4.26/','4.27/']: 
            
            if ((r in ['4.14/','4.15/','4.16/','4.17/','4.18/','4.19/','4.20/','4.27/']
                and featPath == '../features/cnfs_1500-2500/')
                or (r in ['4.14/','4.15/','4.16/','4.17/','4.18/','4.19/','4.20/','4.27/']
                and featPath == '../features/cnfs_500-1000/')
                or (r in ['4.23/','4.24/','4.25/','4.26/','4.27/']
                and featPath == '../features/n3000/')):
                continue
            
            weibH = 0
            p_better = 0
            files = os.listdir(featPath + r)
            for feat in files:
                dist = feat.replace('cnf.feat','dist')
                
                rand = random.uniform(0,1)
                
                if os.path.isfile(distPath + dist):     
                    # open distfile to get the best fitted distribution
                    with open(distPath + dist, 'r') as f:
                        # This file contains the data for the maximum likelihood estimates.
                        lines = f.readlines()

                        p_better += 1
                        loc = int(lines[1].split()[1])
                        scale = float(lines[2].split()[1])
                        shape = float(lines[5].split()[1])
                        
                        features = None                        
                        # open the feature file to get the feature vector
                        with open(featPath + r + feat, 'r') as fl:
                            fl.readline()
                            line = fl.readline()
                            features = list(map(float,line.split(','))) 
                            # Only consider the features with a minimum degree of variance.
                            # These features were obtained by first scaling them to a minimum of zero and a maximum of one.
                            # Then, the features with a variance (after scaling) of at least 0.05 were selected.
                            # Four additional features were also added.
                            # The handpicked features are [25, 41, 46, 55].
                            features = [features[i] for i in [ 0, 1, 2, 3, 13, 15, 16, 22, 24, 25, 32, 34, 38, 40, 41, 43, 46, 53, 54, 55, 56, 57, 58, 59, 60, 61, 66, 68, 69, 70, 71, 72, 73, 78]]
                            
                        # open the output file to get the runtimes
                        

                        X= []
                        Y= []

                        X.append(features)
                        Y.append([numpy.log(loc)])
                        feats_labels.append((dist, X, Y))
    return feats_labels

def normalize(data, mean = [], var = []):
    """
        Normalizes the data to have (approx.) mean 0 and variance 1.
    """
    columns = len(data[0])
    normalizedData = []
    if len(mean) == 0:
        mean = numpy.mean(data,axis=0)

    normalizedData = [[x1-x2 for (x1,x2) in zip(x3,mean)] for x3 in data]
    if len(var) == 0:
        var = numpy.var(data,axis=0)

    normalizedData = [[x1/numpy.sqrt(x2) if x2 != 0 else 0 for (x1,x2) in zip(x3,var)] for x3 in normalizedData]

    return normalizedData, mean, var

def sample(data, k=0.1, seed=1909):
    """
        Used for cross-validation.
    """
    random.seed(seed)
    random.shuffle(data)
    resultX = []
    resultY=[]
    p = 1
    n = len(data)
    while p*k <= 1: 
        X = [x[1] for x in data[int((p-1)*k*n):int(p*k*n)]]
        Y = [x[2] for x in data[int((p-1)*k*n):int(p*k*n)]]
        resultX.append(X)
        resultY.append(Y)
        p += 1
    return resultX, resultY
    
