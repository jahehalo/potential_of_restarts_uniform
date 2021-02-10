import numpy as np
import collections as coll

outPercentage = 1



def insertIntoList(l,x):
    index = 0
    for elem in l:
        if elem[1] >= x[1]:
            l.insert(index,x)
            return l
        index += 1
        
    l.insert(index,x)
    return l
            
        
def findValueOfInstance(l,instance):
    for elem in l:
        if elem[0] == instance:
            return elem[1]
            
    return None



with open("./nohup2.out",'r') as nohup:

  
    results_val_rmse_shape = []
    results_val_log_rmse_scale = []
    results_val_loss = []
    results_val_metric_weib = []
  
    abs_min_val_rmse_shape = 1000
    abs_min_val_log_rmse_scale = 1000
    abs_min_val_loss = 1000
    abs_min_val_metric_weib = 1000
    
    total_val_rmse_shape = {}
    total_val_log_rmse_scale = {}
    total_val_loss = {}
    total_val_metric_weib = {}
    
    median_val_rmse_shape = {}
    median_val_log_rmse_scale = {}
    median_val_loss = {}
    median_val_metric_weib = {}

    metric = {}

    max_id = 0
    line = nohup.readline()
    while(line != ""):
        
        
        if line.split("_")[0] != "./models/log":
            line = nohup.readline()
            continue
            
            
            
        line = line.split(" : ")
        
        instance = line[0]
        i = int(instance.split("_id_")[1].split(".")[0])
        if i > max_id:
            max_id = i

        instance = instance[:(instance.find("_id_"))]
        data = line[1].strip()
        data = data.split(", ")
        data = np.array(data)
        data = data.astype(np.float)   
            
        if instance not in metric:
            metric[instance] = [0.0, 0.0, 0.0]

        metric[instance] = list(map((lambda x, y: x+y), metric[instance], data))
        quality = 0
        line = nohup.readline()

    max_id = max_id + 1
    for key in metric:
        metric[key] = list(map(lambda x: x/max_id, metric[key]))
    metricVal = coll.OrderedDict(sorted(metric.items(),key = lambda t: t[1][0]))
    metricShape = coll.OrderedDict(sorted(metric.items(),key = lambda t: t[1][1]))
    metricScale = coll.OrderedDict(sorted(metric.items(),key = lambda t: t[1][2]))
    print("sorted by val_loss")
    for key in metricVal:
        print(key, " : ", metricVal[key])

    print("\nsorted by shape")
    for key in metricShape:
        print(key, " : ", metricShape[key])

    print("\nsorted by scale")
    for key in metricScale:
        print(key, " : ", metricScale[key])
