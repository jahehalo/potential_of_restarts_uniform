import _pickle
import os
import os.path
import random
import numpy
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
    trainingSetRatio = 0.9
    
    #weibtraining = []
    #weibtrainingLabels = []
    weibtraining = []
    weibtrainingLabels = []
    weibDict = {}
    
    weibtesting = []
    weibtestingLabels = []
    
    logtraining = []
    logtrainingLabels = []
    logDict = {}
    
    logtesting = []
    logtestingLabels = []
    
    partraining = []
    partrainingLabels = []
    parDict = {}
    
    partesting = []
    partestingLabels = []
    
    
    
    
    trainingInstances = []
    testing = []
    testingLabels = []
    testingInstances = []
    
    for featPath in [ '../features/cnfs_1500-2500/']: #, ,'../features/n3000/', '../features/cnfs_500-1000/',
    
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
                dist = feat.replace('cnf.feat','dist')
                
                rand = random.uniform(0,1)
                
                if os.path.isfile(distPath + dist):
                        
                        
                    # open distfile to get the best fitted distribution
                    with open(distPath + dist, 'r') as f:
                        d = f.readline().split()[1]
                        if rand <= trainingSetRatio:
                            weibapp = []
                            logapp = []
                            parapp = []                            
                            loc = int(f.readline().split()[1])
                            if d == 'weibull':
                                features = None
                                f.readline()
                                f.readline()
                                f.readline()
                                shape = float(f.readline().split()[1])
                            
                                # open the feature file to get the feature vector
                                with open(featPath + r + feat, 'r') as fl:
                                    fl.readline()
                                    line = fl.readline()
                                    features = list(map(float,line.split(','))) 
                                    
                                # open the output file to get the runtimes
                                
                                with open(featPath.replace('features/','outputs/') + 'Folder/' + r + feat.replace('feat','result')) as fl:
                                    runtimes = fl.readlines()
                                a= []
                                for run in runtimes:
                                    a.append(int(run)-loc)
                                a.sort()
                                i = 1
                                l = len(a)
                                for run in a:
                                    weibtraining.append(features)
                                    weibtrainingLabels.append([run,i/l,shape,dist])
                                    i = i+1
                                    
                            elif d == 'lognorm':
                             
                                features = None
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                shape = float(f.readline().split()[1])
 
                                # open the feature file to get the feature vector
                                with open(featPath + r + feat, 'r') as fl:
                                    fl.readline()
                                    line = fl.readline()
                                    features = list(map(float,line.split(',')))
 
                                # open the output file to get the runtimes
 
                                with open(featPath.replace('features/','outputs/') + 'Folder/' + r + feat.replace('feat','result')) as fl:
                                    runtimes = fl.readlines()
                                a= []
                                for run in runtimes:
                                    a.append(int(run))
                                a.sort()
                                i = 1
                                l = len(a)
                                for run in a:
                                    logtraining.append(features)
                                    logtrainingLabels.append([run,i/l,shape,dist])
                                    i = i+1
                                    
                            if d == 'pareto':
                            
                                features = None
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                shape = float(f.readline().split()[1])
                            
                                # open the feature file to get the feature vector
                                with open(featPath + r + feat, 'r') as fl:
                                    fl.readline()
                                    line = fl.readline()
                                    features = list(map(float,line.split(','))) 
                                    
                                # open the output file to get the runtimes
                                
                                with open(featPath.replace('features/','outputs/') + 'Folder/' + r + feat.replace('feat','result')) as fl:
                                    runtimes = fl.readlines()

                                a= []
                                for run in runtimes:
                                    a.append(int(run))#-loc)
                                a.sort()
                                i = 1
                                l = len(a)

                                for run in a:
                                    partraining.append(features)
                                    partrainingLabels.append([int(run),i/l,shape,dist])
                                    i += 1
                    ####### ab hier: test
                        else:
                                     
                            weibapp = []
                            logapp = []
                            parapp = []
                            
                            loc = int(f.readline().split()[1])
                        
                            if d == 'weibull':
                            
                                with open(featPath + r + feat, 'r') as fl:
                                    fl.readline()
                                    line = fl.readline()
                                    features = list(map(float,line.split(',')))
                            
                                with open(featPath.replace('features/','outputs/') + 'Folder/' + r + feat.replace('feat','result')) as fl:
                                    runtimes = fl.readlines()

                            
                                scale = numpy.float(f.readline().split()[1])
                                f.readline()
                                f.readline()
                                shape = float(f.readline().split()[1])

                                a= []
                                for run in runtimes:
                                    a.append(int(run)-loc)

                                a.sort()
                                i = 1
                                l = len(a)
                                for run in a:
                                    weibtesting.append(features)
                                    #weibapp.append([run, i/l, shape, scale, loc, dist])
                                    weibtestingLabels.append([run, i/l, shape, scale, loc, dist])
                                    i += 1


                                
                                #weibtestingLabels.append(weibapp)
                            
                            elif d == 'lognorm':
                            
                                with open(featPath + r + feat, 'r') as fl:
                                    fl.readline()
                                    line = fl.readline()
                                    logtesting.append(list(map(float,line.split(','))))
                            
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                logapp.append(float(f.readline().split()[1]))
                                f.readline()
                                f.readline()
                                logapp.append(float(f.readline().split()[1]))
                                
                                logtestingLabels.append(logapp)
                            elif d == 'pareto':
                            
                                with open(featPath + r + feat, 'r') as fl:
                                    fl.readline()
                                    line = fl.readline()
                                    partesting.append(list(map(float,line.split(','))))
                            
                                parapp.append(loc)
                            
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                tmp = float(f.readline().split()[1])
                                f.readline()
                                f.readline()
                                parapp.append(numpy.log(numpy.float(f.readline().split()[1])))
                                parapp.append(tmp)
                                
                                partestingLabels.append(parapp)
                            
                                    
                                    
    return weibtraining, weibtrainingLabels, logtraining, logtrainingLabels, partraining, partrainingLabels, weibtesting, weibtestingLabels, logtesting, logtestingLabels, partesting, partestingLabels
                                    

def createData():
    trainingSetRatio = 0.8
    
    weibtraining = []
    weibtrainingLabels = []
    weibDict = {}
    
    weibtesting = []
    weibtestingLabels = []
    
    logtraining = []
    logtrainingLabels = []
    logDict = {}
    
    logtesting = []
    logtestingLabels = []
    
    partraining = []
    partrainingLabels = []
    parDict = {}
    
    partesting = []
    partestingLabels = []
    
    
    
    
    trainingInstances = []
    testing = []
    testingLabels = []
    testingInstances = []
    
    for featPath in [ '../features/cnfs_1500-2500/']: #, ,'../features/n3000/', '../features/cnfs_500-1000/',
    
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
                dist = feat.replace('cnf.feat','dist')
                
                rand = random.uniform(0,1)
                
                if os.path.isfile(distPath + dist):
                   # with open(featPath + r + feat, 'r') as f:
                   #     f.readline()
                   #     line = f.readline()
                   #     
                   #     if rand <= trainingSetRatio:
                   #         trainingInstances.append(feat)
                   #         training.append(list(map(float,line.split(','))))
                   #     else:
                   #         testingInstances.append(feat)
                   #         testing.append(list(map(float,line.split(','))))
                        
                        
                    with open(distPath + dist, 'r') as f:
                        d = f.readline().split()[1]
                        
                        if rand <= trainingSetRatio:
                            weibapp = []
                            logapp = []
                            parapp = []
                            
                            loc = numpy.log(numpy.float(f.readline().split()[1]))
                            
                            
                            
                        
                            if d == 'weibull':
                            
                                with open(featPath + r + feat, 'r') as fl:
                                    fl.readline()
                                    line = fl.readline()
                                    weibtraining.append(list(map(float,line.split(','))))
                                    
                                    
                            
                                weibapp.append(loc)
                            
                                weibapp.append(numpy.float(f.readline().split()[1]))
                                f.readline()
                                f.readline()
                                weibapp.append(float(f.readline().split()[1]))
                                
                                weibtrainingLabels.append(weibapp)
                                
                                with open(featPath.replace('features','outputs') + 'Folder/' + r + feat.replace('feat','result'), 'r') as runs:
                                        out = runs.readlines()
                                        weibDict[str(weibapp)] = out
                            
                            elif d == 'lognorm':
                             
                                features = None
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                shape = float(f.readline().split()[1])
 
                                # open the feature file to get the feature vector
                                with open(featPath + r + feat, 'r') as fl:
                                    fl.readline()
                                    line = fl.readline()
                                    features = list(map(float,line.split(',')))
 
                                # open the output file to get the runtimes
 
                                with open(featPath.replace('features/','outputs/') + 'Folder/' + r + feat.replace('feat','result')) as fl:
                                    runtimes = fl.readlines()
                                a= []
                                for run in runtimes:
                                    a.append(int(run))
                                a.sort()
                                i = 1
                                l = len(a)
                                for run in a:
                                    logtraining.append(features)
                                    logtrainingLabels.append([run,i/l,shape,dist])
                                    i = i+1

                            elif d == 'pareto':
                            
                                with open(featPath + r + feat, 'r') as fl:
                                    fl.readline()
                                    line = fl.readline()
                                    partraining.append(list(map(float,line.split(','))))
                                    
                                    
                            
                                parapp.append(loc)
                            
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                tmp = float(f.readline().split()[1])
                                f.readline()
                                f.readline()
                                parapp.append(numpy.log(numpy.float(f.readline().split()[1])))
                                parapp.append(tmp)
                                
                                partrainingLabels.append(parapp)
                            
                                with open(featPath.replace('features','outputs') + 'Folder/' + r + feat.replace('feat','result'), 'r') as runs:
                                        out = runs.readlines()
                                        parDict[str(parapp)] = out
                        
                         
                            
                            
                        else:
                                     
                            weibapp = []
                            logapp = []
                            parapp = []
                            
                            loc = numpy.log(numpy.float(f.readline().split()[1]))
                            
                            
                            
                        
                            if d == 'weibull':
                            
                                with open(featPath + r + feat, 'r') as fl:
                                    fl.readline()
                                    line = fl.readline()
                                    weibtesting.append(list(map(float,line.split(','))))
                            
                                weibapp.append(loc)
                            
                                weibapp.append(numpy.float(f.readline().split()[1]))
                                f.readline()
                                f.readline()
                                weibapp.append(float(f.readline().split()[1]))
                                
                                weibtestingLabels.append(weibapp)
                            
                            elif d == 'lognorm':
                            
                                with open(featPath + r + feat, 'r') as fl:
                                    fl.readline()
                                    line = fl.readline()
                                    logtesting.append(list(map(float,line.split(','))))
                            
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                logapp.append(float(f.readline().split()[1]))
                                f.readline()
                                f.readline()
                                logapp.append(float(f.readline().split()[1]))
                                
                                logtestingLabels.append(logapp)
                            elif d == 'pareto':
                            
                                with open(featPath + r + feat, 'r') as fl:
                                    fl.readline()
                                    line = fl.readline()
                                    partesting.append(list(map(float,line.split(','))))
                            
                                parapp.append(loc)
                            
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                f.readline()
                                tmp = float(f.readline().split()[1])
                                f.readline()
                                f.readline()
                                parapp.append(numpy.log(numpy.float(f.readline().split()[1])))
                                parapp.append(tmp)
                                
                                partestingLabels.append(parapp)
                            
        
        
    glob_weibDict = weibDict
    glob_logDict = logDict
    glob_parDict = parDict
    
    return weibtraining, weibtrainingLabels, weibDict, logtraining, logtrainingLabels, logDict, partraining, partrainingLabels, parDict, weibtesting, weibtestingLabels, logtesting, logtestingLabels, partesting, partestingLabels
    
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
    
