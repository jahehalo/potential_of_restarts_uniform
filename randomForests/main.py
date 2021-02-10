from sklearn.ensemble import RandomForestClassifier
import sklearn.base
import functions as f
import sys
import getopt
import numpy.random as rand
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
from sklearn.metrics import recall_score
import numpy as np

"""
    This file trains random forests and evaluates them via cross validation.
"""

n_estimators = 50
criterion = "entropy"


kFold = 10
iterations = 10
inFolder = "./dists"
featFolder = "./feats"
speedupFolder = "./speedups"

seed = rand.randint(9999999)

argv = sys.argv[1:]



rand.seed(seed)


clf_base = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)


wTPR = 0
wSPC = 0
wPPV = 0
wNPV = 0
wFPR = 0
wFDR = 0
wFNR = 0

wACC = 0
wBACC = 0
wF1 = 0

lTPR = 0
lSPC = 0
lPPV = 0
lNPV = 0
lFPR = 0
lFDR = 0
lFNR = 0

lACC = 0
lBACC = 0
lF1 = 0

speedupPot = 0
speedup = 0

for i in range(iterations):

    print("Progress:",i+1,"of",iterations)

    features,labels,scores = f.createData(inFolder, featFolder,speedupFolder)
    permutation = rand.permutation(len(features))
    features = [features[i] for i in permutation]
    labels = [labels[i] for i in permutation]
    scores = [scores[i] for i in permutation]

    for k in range(kFold):
    
    
        wTP = 0
        wTN = 0
        wFP = 0
        wFN = 0
    
        lTP = 0
        lTN = 0
        lFP = 0
        lFN = 0

        speedupTmp = 0
        speedupPotTmp = 0

        clf = sklearn.base.clone(clf_base)

        bottom = int(k*len(features)/kFold)
        top = int((k+1)*len(features)/kFold)
    
    
    
        testingFeatures = features[bottom:top]
        testingLabels = labels[bottom:top]
        testingScores = scores[bottom:top]
        trainingFeatures = features[0:bottom]
        trainingFeatures += features[top:]
        trainingLabels = labels[0:bottom]
        trainingLabels += labels[top:]
    
        #print(trainingFeatures)
        #print(trainingLabels)
    
        clf = clf.fit(trainingFeatures,trainingLabels)
        results = clf.predict(testingFeatures)
   

   
    
        for i in range(len(results)):
            if testingLabels[i] == "weibull" and results[i] == "weibull":
                wTP += 1
                lTN += 1
                speedupTmp += testingScores[i][0]
                speedupPotTmp += testingScores[i][0]
            elif testingLabels[i] == "weibull" and results[i] == "lognorm":
                wFN += 1
                lFP += 1
                #print(len(testingScores[i]))
                speedupTmp += testingScores[i][1]
                speedupPotTmp += testingScores[i][0]
            elif testingLabels[i] == "lognorm" and results[i] == "weibull":
                wFP += 1
                lFN += 1
                speedupTmp += testingScores[i][0]
                speedupPotTmp += testingScores[i][1]
            elif testingLabels[i] == "lognorm" and results[i] == "lognorm":
                wTN += 1
                lTP += 1
                speedupTmp += testingScores[i][1]
                speedupPotTmp += testingScores[i][1]
            
    
        speedup += speedupTmp/len(results)
        speedupPot += speedupPotTmp/len(results)
    
        wP = wTP + wFN
        wN = wFP + wTN
            
        wTPR += wTP/wP
        wSPC += wTN/wN
        wPPV += wTP/(wTP + wFP)
        wNPV += wTN/(wTN + wFN)
        wFPR += wFP/wN
        wFDR += wFP/(wFP + wTP)
        wFNR += wFN/(wFN + wTP)
    
        wACC += (wTP + wTN)/(wP + wN)
        wBACC += (wTP/wP + wTN/wN)/2
        wF1 += 2*wTP/(2*wTP + wFP + wFN)
    
    
        lP = lTP + lFN
        lN = lFP + lTN
            
        lTPR += lTP/lP
        lSPC += lTN/lN
        lPPV += lTP/(lTP + lFP)
        lNPV += lTN/(lTN + lFN)
        lFPR += lFP/lN
        lFDR += lFP/(lFP + lTP)
        lFNR += lFN/(lFN + lTP)
    
        lACC += (lTP + lTN)/(lP + lN)
        lBACC += (lTP/lP + lTN/lN)/2
        lF1 += 2*lTP/(2*lTP + lFP + lFN)
    
    

        del(clf)


div = kFold*iterations

wTPR = wTPR/div
wSPC = wSPC/div
wPPV = wPPV/div
wNPV = wNPV/div
wFPR = wFPR/div
wFDR = wFDR/div
wFNR = wFNR/div

wACC = wACC/div
wBACC = wBACC/div
wF1 = wF1/div

lTPR = lTPR/div
lSPC = lSPC/div
lPPV = lPPV/div
lNPV = lNPV/div
lFPR = lFPR/div
lFDR = lFDR/div
lFNR = lFNR/div

lACC = lACC/div
lBACC = lBACC/div
lF1 = lF1/div


print("true positive rate:",wTPR,lTPR)
print("true negative rate:",wSPC,lSPC)
print("precision:",wPPV,lPPV)
print("negative prediction value:",wNPV,lNPV)
print("fall-out:",wFPR,lFPR)
print("false discovery rate:",wFDR,lFDR)
print("Miss Rate:",wFNR,lFNR)

print("accuracy:",wACC,lACC)
print("balanced accuracy:",wBACC,lBACC)
print("F1 score:",wF1,lF1)

print("\nSpeedupPot:",speedupPot/div)
print("Speedup:",speedup/div)


#clf = f.loadClf('classifier.clf')




