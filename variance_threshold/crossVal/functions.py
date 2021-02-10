import _pickle
import os
import os.path
import random

# ~ 20% is for Training, ~ 80% is for testing (probabilistic) (see Pareto principle)
def createData():
    
    features = []
    winners = []
    passers = []
    
    
    distFiles = os.listdir("./dists")
    
    for dist in distFiles:
        feat = dist.replace(".dist",".cnf.feat")
        with open("./feats/" + feat,'r') as f:
            f.readline()
            line = f.readline()
            features.append(list(map(float,line.split(','))))
            
        with open("./dists/" + dist, 'r') as f:
            d = f.readline().split(" ")[1]
            winners.append(d)
            content = f.readlines()
            passed = []
            
            if "weibH 0\n" in content:
                passed.append("weibull\n")
            
            if "lognH 0\n" in content:
                passed.append("lognorm\n")
            
            if "gpH 0\n" in content:
                passed.append("pareto\n")
                
            passers.append(passed)
            
            
    return features, winners, passers
            
            
    
    
    

def writeClf(clf, path):
    with open(path,'wb') as f:
        _pickle.dump(clf,f)
        
def loadClf(path):
    with open(path,'rb') as f:
        return _pickle.load(f)
