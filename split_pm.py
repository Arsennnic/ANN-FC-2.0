import tensorflow as tf
from tensorflow import keras
import csv
import pandas as pd
import time
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.image import NonUniformImage

def NN_SPLIT_data_generation(data_file, scale = False, 
        feature_scale = None, target_scale = None, 
        skip_header = 0, trans = None):

    data = np.genfromtxt(data_file,delimiter=',', skip_header = skip_header)
    
    # ncomp, pressure, Fv, ncomp 
    ncomp = int((data.shape[1] - 2) / 2)
    
    feature0 = data[:, 0:(ncomp + 1)]
    
    target_begin = ncomp + 1
    ntarget = 1 + ncomp
    target0 = data[:,target_begin:]

    feature = []
    target = []
    for f, t in zip(feature0, target0):
        flag = True

        if t[0] < 1e-6 or t[0] > 1.0 - 1e-6:
            flag = False

        for i in range(ntarget - 1):
            if t[i + 1] < 1e-30:
                flag = False

        if flag:
            feature.append(f)
            target.append(t)

    feature = np.array(feature)
    target = np.array(target)

    if scale:
        nc = target.shape[1]
        nr = target.shape[0]

        for i in range(nc):
            max_v = target[:,i].max()
            min_v = target[:,i].min()

            if target_scale is not None:
                target_scale.append(min_v)
                target_scale.append(max_v)

            if i == 0:
                target[:,i] = NN_SPLIT_data_transformation(target[:,i], 
                        min_v, max_v)
            else:
                target[:,i] = NN_SPLIT_data_transformation(target[:,i], 
                        min_v, max_v, method = trans)

        nc = feature.shape[1]
        nr = feature.shape[0]

        for i in range(nc):
            max_v = feature[:,i].max()
            min_v = feature[:,i].min()

            if feature_scale is not None:
                feature_scale.append(min_v)
                feature_scale.append(max_v)

            feature[:,i] = NN_SPLIT_data_transformation(feature[:,i], 
                        min_v, max_v)

    return {'feature': feature, 'target': target}


def NN_SPLIT_data_transformation(data, data_min, data_max, 
        method = None):
    """
        1. Default: min-max 
        2. Log: log
        3. New: test new transformation
    """

    if method is None:
        data = (data - data_min) / (data_max - data_min)
    elif method is "log":
        data = np.log(data)
    elif method is "new":
        data = (np.log(data) - np.log(data_min)) / (np.log(data_max) - np.log(data_min))
        data = np.sqrt(data)

    return data



def NN_SPLIT_data_detransformation(data, data_min, data_max,
        method = None):
    """
        1. Default: min-max
        2. Log
        3. new transformation
    """
    if method is None:
        data = data * (data_max - data_min) + data_min
    elif method is "log":
        data = np.exp(data)
    elif method is "new":
        data = np.square(data)
        data = data * (np.log(data_max) - np.log(data_min)) + np.log(data_min)
        data = np.exp(data)

    return data


def NN_SPLIT_scale_target(data, scale, method = None):
    """
        target scale
    """
    n = data.shape[1]

    data[:,0] = NN_SPLIT_data_transformation(data[:,0], scale[0], scale[1])

    for i in range(n - 1):
        data[:,i+1] = NN_SPLIT_data_transformation(data[:,i+1], 
                scale[(i+1)*2], scale[(i+1)*2+1], method = method)

def NN_SPLIT_scale_back_target(data, scale, method = None):
    """
        back to target scale
    """
    n = data.shape[1]

    data[:,0] = NN_SPLIT_data_detransformation(data[:,0], scale[0], scale[1])

    for i in range(n - 1):
        data[:,i+1] = NN_SPLIT_data_detransformation(data[:,i+1], 
                scale[(i+1)*2], scale[(i+1)*2+1], method = method)



def NN_SPLIT_convert_data(data_file, feature, target, pred, fit = 'Fv'):
    data = pd.read_csv(data_file)
    
    nc = data.shape[1]
    nr = data.shape[0]
    
    ncomp = int((nc - 3) / 2)

    X = []
    Y = []
    Z_target = []
    Z_pred = []
    
    count = 0
    for i in range(nr):
        flag = False
        
        if fit is 'Fv':
            Fv = data['Fv'][i]
            if Fv >= 0.9999 or Fv <= 0.0001:
                
                if_insert = True
                for aa in Y:
                    if (np.fabs(data['Pressure'][i] - aa) < 1e-4):
                        if_insert = False

                if if_insert:
                    Y.append(data['Pressure'][i])

                if_insert = True
                for aa in X:
                    if (np.fabs(data['Component 1'][i] - aa) < 1e-4):
                        if_insert = False

                if if_insert:
                    X.append(data['Component 1'][i])

                
                #if data['Pressure'][i] not in Y:
                #    Y.append(data['Pressure'][i])
            
                #if data['Component 1'][i] not in X:
                #    X.append(data['Component 1'][i])
                
                flag = True 
        else:
            K = data[fit][i]
            
            if K <= 1e-15:
                if_insert = True
                for aa in Y:
                    if (np.fabs(data['Pressure'][i] - aa) < 1e-4):
                        if_insert = False

                if if_insert:
                    Y.append(data['Pressure'][i])

                if_insert = True
                for aa in X:
                    if (np.fabs(data['Component 1'][i] - aa) < 1e-4):
                        if_insert = False

                if if_insert:
                    X.append(data['Component 1'][i])

                #if data['Pressure'][i] not in Y:
                #    Y.append(data['Pressure'][i])
            
                #if data['Component 1'][i] not in X:
                #    X.append(data['Component 1'][i])
                
                flag = True
        
        f = feature[count]
        t = target[count]
        p = pred[count]
        
        if_insert = True
        for aa in X:
            if (np.fabs(f[0] - aa) < 1e-4):
                if_insert = False

        if if_insert:
            X.append(f[0])

        if_insert = True
        for aa in Y:
            if (np.fabs(f[-1] - aa) < 1e-4):
                if_insert = False

        if if_insert:
            Y.append(f[-1])

        #if f[0] not in X:
        #    X.append(f[0])
        #if f[-1] not in Y:
        #    Y.append(f[-1])
        
        #print(X,Y)

        if flag:
            Z_target.append(-1.0)
            Z_pred.append(-1.0) 
        else:  
            Z_target.append(t)
            Z_pred.append(p)
        
        count += 1
  
    dim = [len(Y), len(X)]
    
    #print(X, Y)
    #print(np.array(Z_target).reshape(dim), np.array(Z_pred).reshape(dim))
    
    return np.array(X), np.array(Y), np.array(Z_target).reshape(dim), np.array(Z_pred).reshape(dim)


# In[11]:


def NN_SPLIT_scale_feature(data, scale):
    """
        scale feature: min-max
    """
    n = data.shape[1]

    for i in range(n):
        data[:,i] = NN_SPLIT_data_transformation(data[:,i], 
                scale[i*2], scale[i*2+1])


def NN_SPLIT_plot(data_file, feature, target, prediction, 
                  fit = 'Fv', plot_name = None):
    
    C_list, P_list, value_target, value_pred = NN_SPLIT_convert_data(data_file, feature, 
                                                                     target, prediction, 
                                                                     fit = fit)
    C_min = np.amin(C_list)
    C_max = np.amax(C_list)
    P_min = np.amin(P_list)
    P_max = np.amax(P_list)
    
    if fit is 'Fv':
        value_target = np.ma.masked_outside(value_target, 0.0001, 0.9999)
        value_pred = np.ma.masked_outside(value_pred, 0.0001, 0.9999)
    else:
        log_target = np.log(value_target)
        log_pred = np.log(value_pred)
        value_target = np.ma.masked_inside(log_target, 0.0, 1e-7)
        value_pred = np.ma.masked_inside(log_target, 0.0, 1e-7)
    
    #print(Fv_target)
    #print(value_pred)  
    
    file_name = None
        
    if plot_name is None:
        plot_name = "ANN-SPLIT-PM-phase"
    
    fig, ax = plt.subplots() 
    im = NonUniformImage(ax, extent=(C_min, C_max, P_min, P_max))
    im.set_cmap("jet")
    im.set_data(C_list, P_list, value_target)
    ax.images.append(im)
    ax.set_xlim(C_min, C_max)
    ax.set_ylim(P_min, P_max)
    ax.set_xlabel("Composition")
    ax.set_ylabel("Pressure, atm")
    #ax.set_title("Flash Calculation")
    fig.colorbar(im, ax = ax)
    
    file_name = plot_name + "-target" + ".eps"    
    plt.savefig(file_name)    
    file_name = plot_name + "-target" + ".pdf"
    plt.savefig(file_name)
    plt.show()
    
    fig, ax = plt.subplots()   
    im = NonUniformImage(ax, extent=(C_min, C_max, P_min, P_max))
    im.set_cmap("jet")
    im.set_data(C_list, P_list, value_pred)
    ax.images.append(im)
    ax.set_xlim(C_min, C_max)
    ax.set_ylim(P_min, P_max)
    ax.set_xlabel("Composition")
    ax.set_ylabel("Pressure, atm")
    #ax2.set_title("Flash Calculation")
    fig.colorbar(im, ax = ax)
    
    file_name = plot_name + "-pred" + ".eps"    
    plt.savefig(file_name)    
    file_name = plot_name + "-pred" + ".pdf"
    plt.savefig(file_name)
    plt.show()
    
    fig, ax = plt.subplots()   
    im = NonUniformImage(ax, extent=(C_min, C_max, P_min, P_max))
    im.set_cmap("jet")
    im.set_data(C_list, P_list, (value_target - value_pred))
    ax.images.append(im)
    ax.set_xlim(C_min, C_max)
    ax.set_ylim(P_min, P_max)
    ax.set_xlabel("Composition")
    ax.set_ylabel("Pressure, atm")
    #ax3.set_title("Flash Calculation")
    fig.colorbar(im, ax = ax)
    
    file_name = plot_name + "-error" + ".eps"    
    plt.savefig(file_name)    
    file_name = plot_name + "-error" + ".pdf"
    plt.savefig(file_name)
    plt.show()


def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.yscale('log')
    plt.ylabel('Loss')

    loss = []
    epoch = []
    val_loss = []

    for e, l, vl in zip(history.epoch, history.history['loss'], history.history['val_loss']):
        if (e > 100):
            epoch.append(e)
            loss.append(l)
            val_loss.append(vl)

    plt.plot(epoch, loss, label='Train Loss')
    plt.plot(epoch, val_loss, label='Train Loss')
    plt.legend()
    plt.show()


class EarlyStoppingByGL(keras.callbacks.Callback):
    def __init__(self, alpha = 0.1, min_epoch = 1000, verbose=0):
        super(keras.callbacks.Callback, self).__init__()
        self.min_val_loss = 1.0
        self.min_val_loss_batch = 1.0
        self.min_epoch = min_epoch
        self.verbose = verbose
        self.GL = 0.0
        self.alpha = alpha
        self.epoch_opt = 0

    def on_epoch_end(self, epoch, logs={}):
        val_loss = logs.get('val_loss')

        if self.min_val_loss > val_loss:
            self.min_val_loss = val_loss

        if (epoch + 1) % 100 == 0:
            self.min_val_loss_batch = 1.0

        if self.min_val_loss_batch > val_loss:
            self.min_val_loss_batch = val_loss
            self.epoch_opt = epoch

        if (epoch + 1) % 100 == 0:
            self.GL = self.min_val_loss_batch / self.min_val_loss - 1.0
            print("    Epoch %05d: MGL: (%1.4f / %1.4f) - 1.0 = %1.4f" 
                    % (epoch + 1, self.min_val_loss_batch, self.min_val_loss, self.GL))

        if (epoch > self.min_epoch and (epoch + 1) % 100 == 0):
            if self.GL > self.alpha:
                print("    Earlystopping! Best Performance epoch %d" %(self.epoch_opt))
                self.model.stop_training = True
        

def NN_train_phase_split_calculation(train_data, test_data, 
                                     hidden_layer = 1, hidden_cells = [10], 
                                     batch_size = 30, epoch = 100, 
                                     validation_split = 0.1, has_val_data = False, validation_data = None,
                                     GL = 0.1, min_epoch = 1000,
                                     plot = False, plot_name = None):

    feature = train_data['feature'] 
    target = train_data['target'] 

    feature_test = test_data['feature'] 
    target_test = test_data['target'] 

    if has_val_data:
        feature_val = validation_data['feature'] 
        target_val = validation_data['target'] 

    nfeature = len(feature[0])
    nstatus = len(target[0])

    model = keras.Sequential()
    model.add(keras.layers.Dense(hidden_cells[0], 
                activation = tf.nn.softmax,
                input_shape = (nfeature,)))

    for i in range(hidden_layer - 1):
        model.add(keras.layers.Dense(hidden_cells[i+1],
                activation = tf.nn.softmax))

    model.add(keras.layers.Dense(nstatus))


    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse', optimizer = optimizer, metrics=['mae'])

    earlystop = EarlyStoppingByGL(alpha = GL, min_epoch = min_epoch)

    filepath = './split_weights'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, 
            monitor='val_loss', verbose = 0, 
            save_best_only=True, save_weights_only=True,
            mode='min')

    if has_val_data:
        history = model.fit(feature, target, batch_size = batch_size, 
                epochs = epoch, validation_data=(feature_val, target_val), 
                verbose=0,
                callbacks = [earlystop, checkpoint])
    else:
        history = model.fit(feature, target, batch_size = batch_size, 
                epochs = epoch, validation_split=validation_split, verbose=0,
                callbacks = [earlystop, checkpoint])

    plot_history(history)

    #model = keras.models.load_model(filepath)
    model.load_weights(filepath)

    W = []
    b = []
    for layer in model.layers:
        W.append(layer.get_weights()[0])
        b.append(layer.get_weights()[1])

    [loss, mae] = model.evaluate(feature_test, target_test, 
            verbose=0)

    print("    Training Loss: %1.5f, MAE: %1.5f" %(loss, mae))

    test_predictions = model.predict(feature_test)

    return W, b, loss, test_predictions

def plot_test_result(target, predictions, plot_name):
    target_min = target.min(axis=0)
    target_max = target.max(axis=0)
    pred_min = predictions.min(axis=0)
    pred_max = predictions.max(axis=0)

    v_min = np.minimum(target_min, pred_min)
    v_max = np.maximum(target_max, pred_max)

    nstatus = target.shape[1] 

    plt.plot([v_min[0], v_max[0]], [v_min[0], v_max[0]], 
            lw = 2, c = 'red', zorder = 10, label = "Equal")
    plt.scatter(target[:,0], predictions[:,0], 
            label = "Testing data")
    plt.xlabel('Mole Fraction')
    plt.ylabel('Predictions')
    plt.legend()
    plt.show()
    file_name = plot_name + "-Fv.eps"
    plt.savefig(file_name)    
    file_name = plot_name + "-Fv.pdf"
    plt.savefig(file_name)    

    for i in range(nstatus - 1):
        plt.plot([v_min[i+1], v_max[i+1]], [v_min[i+1], v_max[i+1]], 
                lw = 2, c = 'red', zorder = 10, label = "Equal")
        plt.scatter(target[:,i+1], predictions[:,i+1], 
                label = "Testing data")

        name = 'K' + str(i+1)
        plt.xlabel(name)
        plt.ylabel('Predictions')
        plt.legend()
        plt.show()
        file_name = plot_name + "-" + name + ".eps"
        plt.savefig(file_name)    
        file_name = plot_name + "-" + name + ".pdf"
        plt.savefig(file_name)    



    error = predictions - target

    plt.hist(error[:,0], bins = 50)
    plt.xlabel("Mole Fraction Prediction Error")
    plt.ylabel("Count")
    plt.show()
    file_name = plot_name + "-Fv-error.eps"
    plt.savefig(file_name)    
    file_name = plot_name + "-Fv-error.pdf"
    plt.savefig(file_name)    

    for i in range(nstatus - 1):
        plt.hist(error[:,i+1], bins = 50)
        name = 'K' + str(i+1) + ' Prediction Error'
        plt.xlabel(name)
        plt.ylabel("Count")
        plt.show()
        file_name = plot_name + '-K' + str(i+1) + "-error.eps"
        plt.savefig(file_name)    
        file_name = plot_name + '-K' + str(i+1) + "-error.pdf"
        plt.savefig(file_name)    
    

def train(train_data, test_data, trans = None,
        hidden_layer = 1, hidden_cells = [10], batch_size = 30, epoch = 100, 
        validation_split = 0.1, validation_data = None,
        GL = 0.1, min_epoch = 1000, train_number = 10, 
        plot = False, plot_name = None):

    time_begin = time.time()

    # training featrue and target
    feature_scale = []
    target_scale = []
    train = NN_SPLIT_data_generation(train_data, scale = True, 
                                                feature_scale = feature_scale,
                                                target_scale = target_scale,
                                                trans = trans)
    #print(len(feature[0]), len(target), feature_scale, target_scale)
    print "*** Number of training examples: " + str(len(train['target']))
    
    # testing featrue and target
    test = NN_SPLIT_data_generation(test_data, 
            scale = False, skip_header = 1)
    NN_SPLIT_scale_feature(test['feature'], feature_scale)
    NN_SPLIT_scale_target(test['target'], target_scale, method = trans)
    print "*** Number of testing examples: " + str(len(test['target']))

    validation = None
    has_val_data = False
    if validation_data is not None:
        validation = NN_SPLIT_data_generation(validation_data,
                scale = False, skip_header = 1)

        NN_SPLIT_scale_feature(validation['feature'], feature_scale)
        NN_SPLIT_scale_target(validation['target'], target_scale, 
                method = trans)
        has_val_data = True
        print "*** Number of validation examples: " + str(len(validation['target']))

    loss_opt = 1.0;
    W_opt = None
    b_opt = None
    pred_opt = None

    print "*** Training begins ..."
    for i in range(train_number):
        print("============ TRAINING PROCESS: %d ============ " %(i + 1))
        W, b, loss, pred = NN_train_phase_split_calculation(
                train, test, 
                hidden_layer = hidden_layer, hidden_cells = hidden_cells, 
                batch_size = batch_size, epoch = epoch, 
                validation_split = validation_split, has_val_data = has_val_data, validation_data = validation,
                GL = GL, min_epoch = min_epoch, 
                plot = plot, plot_name = plot_name)
        if loss < loss_opt:
            loss_opt = loss
            W_opt = W
            b_opt = b
            pred_opt = pred

    NN_SPLIT_scale_back_target(test['target'], target_scale,
            method = trans)
    NN_SPLIT_scale_back_target(pred_opt, target_scale,
            method = trans)

    time_end = time.time()
    print("Time cost: %f seconds" %(time_end - time_begin))

    if (plot):
        plot_test_result(test['target'], pred_opt, plot_name)

    return W_opt, b_opt, loss_opt


def NN_SPLIT_load_matrix_bias_from_file(file_name, level):
    input_name = None
    W = [None] * level
    b = [None] * level
    
    for i in range(level):
        input_name = file_name + 'W' + str(i) + '.csv'
        W[i] = np.loadtxt(input_name, delimiter=',')
        
        input_name = file_name + 'b' + str(i) + '.csv'
        b[i] = np.loadtxt(input_name, delimiter=',')
        
        dim = [1, b[i].shape[0]]
        b[i] = b[i].reshape(dim)
        print(b[i].shape, dim)

    return W, b

def NN_SPLIT_write_matrix_bias_to_file(W, b, file_name = None):
    output_name = None
    
    if file_name is None:
        file_name = 'ANN-SPLIT-model-'
    
    level = len(W)
    
    for i in range(level):
        output_name = file_name + 'W' + str(i) + '.csv'
        np.savetxt(output_name, W[i], delimiter = ',')
        
        output_name = file_name + 'b' + str(i) + '.csv'
        np.savetxt(output_name, b[i], delimiter = ',')



