from sklearn.ensemble import RandomForestClassifier
import functions as f


tr,trl,tri,te,tel,tei = f.createData()

clf_r4_24 = f.loadClf('./classifiers/4.clf')
clf_r4_25 = f.loadClf('./classifiers/5.clf')
clf_r4_26 = f.loadClf('./classifiers/6.clf')
clf_r4_27 = f.loadClf('./classifiers/7.clf')

clf_r4_24_e = f.loadClf('./classifiers/8.clf')
clf_r4_25_e = f.loadClf('./classifiers/9.clf')
clf_r4_26_e = f.loadClf('./classifiers/10.clf')
clf_r4_27_e = f.loadClf('./classifiers/11.clf')


plus = 0
minus = 0

logToExp = 0
logToWeib = 0
logToNone = 0

expToLog = 0
expToWeib = 0
expToNone = 0

weibToLog = 0
weibToExp = 0
weibToNone = 0

noneToLog = 0
noneToExp = 0
noneToWeib = 0


for i in range(len(te)):
    p1 = clf_r4_24.predict([te[i]])[0]
    p2 = clf_r4_25.predict([te[i]])[0]
    p3 = clf_r4_26.predict([te[i]])[0]
    p4 = clf_r4_27.predict([te[i]])[0]
    
    ratio = 1/te[i][7]
    
    logc = 0
    weibc = 0
    expc = 0
    nonec = 0
    
    counter = 0.04
    for pred in [p1,p2,p3,p4]:
    
        w = (4.2+counter)/(4.2+counter+50*abs(4.2+counter - ratio))
        
        #print(w)
         
    
        if pred == "exp":
            expc += 1*w
        elif pred == "lognorm":
            logc += 1*w
        elif pred == "weibull":
            weibc += 1*w
        else:
            nonec += 1*w
            
        counter += 0.01
            
    p = "none"    
    
    if logc >= weibc and logc >= expc and logc >= nonec:
        p = "lognorm"
    elif weibc >= expc and weibc >= nonec:
        p = "weibull"
    elif expc >= nonec:
        p = "exp"
    else:
        p = "none"
    
    
    
    
    
    #print("prediction:" + str(p))
    #print("label:" + str(tel[i]))
    if p == tel[i]:
        plus += 1
    else:
        minus += 1
        
        if p == "lognorm":
            if tel[i] == "exp":
                logToExp += 1
            elif tel[i] == "weibull":
                logToWeib += 1
            else:
                logToNone += 1
                
        elif p == "exp":
            if tel[i] == "lognorm":
                expToLog += 1
            elif tel[i] == "weibull":
                expToWeib += 1
            else:
                expToNone += 1
                
        elif p == "weibull":
            if tel[i] == "lognorm":
                weibToLog += 1
            elif tel[i] == "exp":
                weibToExp += 1
            else:
                weibToNone += 1
                
        else:
            if tel[i] == "lognorm":
                noneToLog += 1
            elif tel[i] == "exp":
                noneToExp += 1
            else:
                noneToWeib += 1
                

print("total number of tests: " + str(plus + minus))
print("positive results: " + str(plus))
print("negative results: " + str(minus))
print("ratio of positive results: " + str(plus/(plus + minus)))
print("l2e: " + str(logToExp) + "\n" + 
        "l2w: " + str(logToWeib) + "\n" +
        "l2n: " + str(logToNone) + "\n" +
        "e2l: " + str(expToLog) + "\n" +
        "e2w: " + str(expToWeib) + "\n" +
        "e2n: " + str(expToNone) + "\n" +
        "w2l: " + str(weibToLog) + "\n" +
        "w2e: " + str(weibToExp) + "\n" +
        "w2n: " + str(weibToNone) + "\n" +
        "n2l: " + str(noneToLog) + "\n" +
        "n2e: " + str(noneToExp) + "\n" +
        "n2w: " + str(noneToWeib))


#print("prediction before: ",clf.predict([[0,0],[1,1]]))

#f.writeClf(clf,'./classifiers/11.clf')



#print("prediction after: ",clf.predict([[0,0],[1,1]]))

print("Done")
