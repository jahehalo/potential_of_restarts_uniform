from sklearn.ensemble import RandomForestClassifier
import numpy as N
import functions as f


features, winners, passers = f.createData()
n = len(features)
featuresC = list(features)
winnersC = list(winners)
passersC = list(passers)

weibOddSum = 0
lognOddSum = 0
parOddSum = 0
forestOddSum = 0



foo = open("./averages.pre",'w')
foo.close()
foo = open("./quotients.pre",'w')
foo.close()


iterations = 100
split = 10

shift = int(n/split)

count = 0

for it in range(iterations):

    

    perm = N.random.permutation(n)
    
    

    for i in range(n):
        features[i] = featuresC[perm[i]]
        winners[i] = winnersC[perm[i]]
        passers[i] = passersC[perm[i]]
        
    for sp in range(split):
    
        count += 1
    
        #print("start split", sp+1, "of iteration", it+1)
    
        nweib = 0
        nlogn = 0
        npar = 0
        correct = 0
        incorrect = 0
    
        start = sp * shift;
        end = n if sp == split-1 else (sp+1) * shift
        
        testingFeatures = features[start:end]
        trainingFeatures = list(features)
        trainingFeatures[start:end] = []
        
        testingWinners = winners[start:end]
        trainingWinners = list(winners)
        trainingWinners[start:end] = []
        
        testingPassers = passers[start:end]
        trainingPassers = list(passers)
        trainingPassers[start:end] = []
        
        

        
        
        clf = RandomForestClassifier(n_estimators=100, criterion="entropy")
        clf = clf.fit(trainingFeatures,trainingWinners)
        
        for t in range(len(testingFeatures)):
            pred = clf.predict([testingFeatures[t]])[0]
            
            
            if "weibull\n" in testingPassers[t]:
                nweib += 1
            if "lognorm\n" in testingPassers[t]:
                nlogn += 1
            if "pareto\n" in testingPassers[t]:
                npar += 1
            if pred in testingPassers[t]:
                correct += 1
            else:
                incorrect += 1
            
            
        
            
        tot = correct + incorrect
        
        
        
        weibOddSum += (nweib/tot)/(1-(nweib/tot))
        lognOddSum += (nlogn/tot)/(1-(nweib/tot))
        parOddSum += (npar/tot)/(1-(nweib/tot))
        forestOddSum += (correct/tot)/(1-(correct/tot))
        
        weibTmp = weibOddSum/count
        lognTmp = lognOddSum/count
        parTmp = parOddSum/count
        forestTmp = forestOddSum/count
        
        with open("./averages.pre", 'a') as f:
            f.write(str(weibTmp) + "\n")
            f.write(str(lognTmp) + "\n")
            f.write(str(parTmp) + "\n")
            f.write(str(forestTmp) + "\n\n")
            
        with open("./quotients.pre", 'a') as f:
            f.write(str(forestTmp/weibTmp) + "\n")
            f.write(str(forestTmp/lognTmp) + "\n")
            f.write(str(forestTmp/parTmp) + "\n\n")
        
        
        
div = iterations * split
    
weibOddAv = weibOddSum / div
lognOddAv = lognOddSum / div
parOddAv = parOddSum / div
forestOddAv = forestOddSum / div


#print(weibOddAv)
#print(lognOddAv)
#print(parOddAv)
#print(forestOddAv)        


#print("weib",forestOddAv/weibOddAv)
#print("logn",forestOddAv/lognOddAv)
#print("par",forestOddAv/parOddAv)

with open("./averages.final",'w') as f:
    f.write(str(weibOddAv) + "\n")
    f.write(str(lognOddAv) + "\n")
    f.write(str(parOddAv) + "\n")
    f.write(str(forestOddAv))
    
with open("./quotients.final",'w') as f:
    f.write(str(forestOddAv/weibOddAv) + "\n")
    f.write(str(forestOddAv/lognOddAv) + "\n")
    f.write(str(forestOddAv/parOddAv))
