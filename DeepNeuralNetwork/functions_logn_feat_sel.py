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
from scipy.special import erf
from scipy.special import erfinv

                                    
def createSampleData():
    """
        This function returns the data for all lognormal distributions (according to the higher p-value in the KS test).
    """
    run_list = []
    feats = []
    label = []
    feat_labels = []
    
    for featPath in [ '../features/cnfs_1500-2500/']: 
        # This path contains the features for all training instances.
    
        distPath = '../dists/'
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
                rand = random.uniform(0,1)
                
                if os.path.isfile(distPath + dist):
                        
                        
                    # open distfile to get the best fitted distribution
                    with open(distPath + dist, 'r') as f:
                        lines = f.readlines()

                        if float(lines[9].split()[1]) > float(lines[17].split()[1]):
                            # Only lognormal distributions (according to the higher p-value in the KS test) should be considered.
                            # If this part is exectued, then the Weibull distribution is the better fitting distribution.
                            continue 
                        mu = float(lines[10].split()[1])
                        sigma = float(lines[13].split()[1])
                        
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
                        
                        with open(featPath.replace('features/','outputs/') + 'Folder/' + r + feat.replace('feat','result')) as fl:
                            # This file contains the measured runtimes.
                            runtimes = fl.readlines()
                        a= []
                        for run in runtimes:
                            a.append(int(run))
                        a.sort()
                        i = 1
                        l = len(a)
                        run_list = []
                        X= []
                        Y= []
                        for run in a:
                            run_list.append(a)
                            X.append(features)
                            Y.append([run,i/l,sigma,mu])
                            i = i+1

                        feat_labels.append((run_list, X,Y))
                        
    return feat_labels

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

def sample(data, k=0.1, seed=1909, dist_names=False):
    """
        Used for cross-validation.
    """
    
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
    
