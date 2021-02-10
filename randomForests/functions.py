import _pickle
import os
import os.path
import random


def createData(folder,featFolder,speedupFolder):
    files = os.listdir(folder)
    
    features = []
    labels = []
    scores = []
    
    for filename in files:
        if filename.endswith(".dist"):
            with open(folder + "/" + filename,"r") as f:
                content = f.readlines()
                pWeib = float(content[9].split(" ")[1])
                pLogn = float(content[17].split(" ")[1])
                pPar = float(content[25].split(" ")[1])
                
                if pWeib >= pLogn:
                    labels.append("weibull")
                else:
                    labels.append("lognorm")
                    
            with open(featFolder + "/" + filename.replace(".dist",".cnf.feat"),"r") as f:
                content = f.readlines()
                tmp = list(map(float,content[1].split(',')))
                tmp = [tmp[i] for i in [ 0, 1, 2, 3, 13, 15, 16, 22, 24, 25, 32, 34, 38, 40, 41, 43, 46, 53, 54, 55, 56, 57, 58, 59, 60, 61, 66, 68, 69, 70, 71, 72, 73, 78]]
                features.append(tmp)
                
            with open(speedupFolder + "/" + filename.replace(".dist",".score"),"r") as f:
                content = f.readlines()
                content = [float(x) for x in content]
                scores.append(content)
                
    
    return features, labels, scores
                
    
    

def writeClf(clf, path):
    with open(path,'wb') as f:
        _pickle.dump(clf,f)
        
def loadClf(path):
    with open(path,'rb') as f:
        return _pickle.load(f)
