from keras import backend as K


def rmse_shape_par(y_true, y_pred):
    shapePred = K.dot(K.constant([0,0,1,0],dtype = 'float64',shape = (1,4)),K.transpose(y_true))
    shape = K.dot(K.constant([0,0,1,0],dtype = 'float64',shape = (1,4)),K.transpose(y_pred))
    shape = shape-2
    result = K.mean(K.square(shape-shapePred),axis=-1)
    result = K.sqrt(result)
    return result

def log_rmse_scale_par(y_true, y_pred):
    scalePred = K.dot(K.constant([0,0,0,1],dtype = 'float64',shape = (1,4)),K.transpose(y_true))
    scale = K.dot(K.constant([0,1,0,0],dtype = 'float64',shape = (1,4)),K.transpose(y_pred))
    scale = scale * (10**8)
    result = K.mean(K.square(K.log(scale)-K.log(scalePred)), axis = -1)
    result = K.sqrt(result)
    return result
