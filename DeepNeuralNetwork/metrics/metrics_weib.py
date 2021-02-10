from keras import backend as K

def weib_negative_log_like(runtime,scale,shape):
    div = runtime/scale
    return - K.log(shape)+K.log(scale)  -(shape-1)*(K.log(runtime)-K.log(scale)) + K.pow(div,shape)

def metric_weib(y_true, y_pred):

    runtime = K.dot(K.constant([1,0,0,0,0],dtype = 'float64',shape = (1,5)),K.transpose(y_true))
    shapePred = K.dot(K.constant([0,0,1,0,0],dtype = 'float64',shape = (1,5)),K.transpose(y_true))
    scalePred = K.dot(K.constant([0,0,0,1,0],dtype = 'float64',shape = (1,5)),K.transpose(y_true))
    max_nll = K.dot(K.constant([0,0,0,0,1],dtype = 'float64',shape = (1,5)),K.transpose(y_true))
    

    scale = K.dot(K.constant([0,1,0,0,0],dtype = 'float64',shape = (1,5)),K.transpose(y_pred))
    shape = K.dot(K.constant([0,0,1,0,0],dtype = 'float64',shape = (1,5)),K.transpose(y_pred))
 
    scale = scale * (10**8)
    one = K.ones_like(shape)

    point_max = weib_negative_log_like(runtime, scalePred, shapePred)
    shape_res = shape - shapePred
    pred_nll = weib_negative_log_like(runtime, scale, shape)
    result = (pred_nll-point_max)/max_nll
    return 18.75*K.sum(result, axis=-1)

def rmse_shape_weib(y_true, y_pred):
    shapePred = K.dot(K.constant([0,0,1,0,0],dtype = 'float64',shape = (1,5)),K.transpose(y_true))
    shape = K.dot(K.constant([0,0,1,0,0],dtype = 'float64',shape = (1,5)),K.transpose(y_pred))
    result = K.mean(K.square(shape-shapePred),axis=-1)
    result = K.sqrt(result)
    return result

def log_rmse_scale_weib(y_true, y_pred):
    scalePred = K.dot(K.constant([0,0,0,1,0],dtype = 'float64',shape = (1,5)),K.transpose(y_true))
    scale = K.dot(K.constant([0,1,0,0,0],dtype = 'float64',shape = (1,5)),K.transpose(y_pred))
    scale = scale * (10**8)
    result = K.mean(K.square(K.log(scale)-K.log(scalePred)), axis = -1)
    result = K.sqrt(result)
    return result
