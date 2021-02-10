import _pickle
import os
import os.path
import random


def createData(folder,featFolder):
    files = os.listdir(folder)
    
    features = []
    labels = []
    
    for filename in files:
        if filename.endswith(".score"):
            with open(folder + "/" + filename,"r") as f:
                content = f.readlines()
                weibScore = float(content[0])
                lognScore = float(content[1])
                
                if weibScore / lognScore < 1.1 and lognScore/weibScore < 1.1:
                    labels.append("weibull")
                elif weibScore > lognScore:
                    labels.append("weibull")
                else:
                    labels.append("lognorm")
                
                    
            with open(featFolder + "/" + filename.replace(".score",".cnf.feat"),"r") as f:
                content = f.readlines()
                tmp = list(map(float,content[1].split(',')))
                #tmp = [tmp[i] for i in [ 0, 1, 2, 3, 13, 15, 16, 22, 24, 25, 32, 34, 38, 40, 41, 43, 46, 53, 54, 55, 56, 57, 58, 59, 60, 61, 66, 68, 69, 70, 71, 72, 73, 78]]
                features.append(tmp)
                
    
    return features, labels
                
    
    

def writeClf(clf, path):
    with open(path,'wb') as f:
        _pickle.dump(clf,f)
        
def loadClf(path):
    with open(path,'rb') as f:
        return _pickle.load(f)
