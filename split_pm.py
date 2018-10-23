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

class ANN_SPLIT:
    """
        ANN model for split calculation
    """
    def __init__(self, W = None, b = None, scale = None, 
            file_name = None, level = 0):
        if W is not None and b is not None and scale is not None:
            self.W = W
            self.b = b
            self.scale = scale
        elif file_name is not None and level > 0:
            self.W, self.b, self.scale = read_weights_from_file(file_name, level)

        self.model = load_model(self.W, self.b)
        self.comp = None

    def store_model(self, file_name):
        to_file(self.W, self.b, self.scale, file_name)

    def predict(self, feature):
        pred = predict_split(self.model, self.scale, feature)

        return pred

    def split_test(self, file_name):
        self.comp = compare_prediction(file_name, self.model, self.scale)

    def plot(self, plot_name = None):
        if self.comp is not None:
            plot_test_result(self.comp['target'], 
                    self.comp['prediction'], plot_name)

def get_data_scale(data):
    scale = []

    nc = data.shape[1]
    nr = data.shape[0]

    for i in range(nc):
        max_v = data[:,i].max()
        min_v = data[:,i].min()

        scale.append(min_v)
        scale.append(max_v)

    return scale

def get_data_rescale(scale1, scale2):
    nc = len(scale1) / 2

    rescale = []
    for i in range(nc):
        if scale1[i*2] < scale2[i*2]:
            rescale.append(scale1[i*2])
        else:
            rescale.append(scale2[i*2])

        if scale1[i*2+1] < scale2[i*2+1]:
            rescale.append(scale1[i*2+1])
        else:
            rescale.append(scale2[i*2+1])

    return rescale

def read_data(data_file, scale = False, 
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
                target[:,i] = data_trans(target[:,i], 
                        min_v, max_v)
            else:
                target[:,i] = data_trans(target[:,i], 
                        min_v, max_v, method = trans)

        nc = feature.shape[1]
        nr = feature.shape[0]

        for i in range(nc):
            max_v = feature[:,i].max()
            min_v = feature[:,i].min()

            if feature_scale is not None:
                feature_scale.append(min_v)
                feature_scale.append(max_v)

            feature[:,i] = data_trans(feature[:,i], 
                        min_v, max_v)

    return {'feature': feature, 'target': target}


def data_trans(data, data_min, data_max, 
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



def data_detrans(data, data_min, data_max,
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


def scale_target(data, scale, method = None):
    """
        target scale
    """
    n = data.shape[1]

    data[:,0] = data_trans(data[:,0], scale[0], scale[1])

    for i in range(n - 1):
        data[:,i+1] = data_trans(data[:,i+1], 
                scale[(i+1)*2], scale[(i+1)*2+1], method = method)

def scale_back_target(data, scale, method = None):
    """
        back to target scale
    """
    n = data.shape[1]

    data[:,0] = data_detrans(data[:,0], scale[0], scale[1])

    for i in range(n - 1):
        data[:,i+1] = data_detrans(data[:,i+1], 
                scale[(i+1)*2], scale[(i+1)*2+1], method = method)


def scale_feature(data, scale):
    """
        scale feature: min-max
    """
    n = data.shape[1]

    for i in range(n):
        data[:,i] = data_trans(data[:,i], 
                scale[i*2], scale[i*2+1])

def scale_back_feature(data, scale):
    """
        back to feature scale
    """
    n = data.shape[1]

    for i in range(n):
        data[:,i] = data_detrans(data[:,i], 
                scale[i*2], scale[i*2+1])


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
    plt.plot(epoch, val_loss, label='Validation Loss')
    plt.legend()
    plt.show()

    return [epoch, loss, val_loss]


class EarlyStoppingByGL(keras.callbacks.Callback):
    def __init__(self, alpha = 0.1, min_epoch = 1000, epoch_strip = 100, verbose=0):
        super(keras.callbacks.Callback, self).__init__()
        self.min_val_loss = 1.0
        self.min_val_loss_batch = 1.0
        self.min_epoch = min_epoch
        self.epoch_strip = epoch_strip
        self.verbose = verbose
        self.GL = 0.0
        self.alpha = alpha
        self.epoch_opt = 0

    def on_epoch_end(self, epoch, logs={}):
        val_loss = logs.get('val_loss')

        if self.min_val_loss > val_loss:
            self.min_val_loss = val_loss
            self.epoch_opt = epoch

        if (epoch + 1) % self.epoch_strip == 0:
            self.min_val_loss_batch = 1.0

        if self.min_val_loss_batch > val_loss:
            self.min_val_loss_batch = val_loss

        if (epoch + 1) % self.epoch_strip == 0:
            self.GL = self.min_val_loss_batch / self.min_val_loss - 1.0
            print("    Epoch %05d: MGL: (%1.4f / %1.4f) - 1.0 = %1.4f" 
                    % (epoch + 1, self.min_val_loss_batch, self.min_val_loss, self.GL))

        if (epoch > self.min_epoch and (epoch + 1) % self.epoch_strip == 0):
            if self.GL > self.alpha:
                print("    Earlystopping! Best Performance epoch %d" %(self.epoch_opt))
                self.model.stop_training = True
        

def train_model(train_data, test_data, 
                                     hidden_layer = 1, hidden_cells = [10], 
                                     batch_size = 30, epoch = 100, 
                                     validation_split = 0.1, has_val_data = False, validation_data = None,
                                     GL = 0.1, GL_epoch_strip = 100, min_epoch = 1000,
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

    earlystop = EarlyStoppingByGL(alpha = GL, min_epoch = min_epoch,
            epoch_strip = GL_epoch_strip)

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

    loss_history = plot_history(history)

    #model = keras.models.load_model(filepath)
    model.load_weights(filepath)

    W = []
    b = []
    for layer in model.layers:
        W.append(layer.get_weights()[0])
        b.append(layer.get_weights()[1])

    [loss, mae] = model.evaluate(feature_test, target_test, 
            verbose=0)

    print("    Testing Loss: %1.5f, MAE: %1.5f" %(loss, mae))

    test_predictions = model.predict(feature_test)

    return W, b, loss, test_predictions, loss_history

def plot_test_result(target, predictions, plot_name):
    target_min = target.min(axis=0)
    target_max = target.max(axis=0)
    pred_min = predictions.min(axis=0)
    pred_max = predictions.max(axis=0)

    v_min = np.minimum(target_min, pred_min)
    v_max = np.maximum(target_max, pred_max)

    nstatus = target.shape[1] 

    plt.clf()
    plt.plot([v_min[0], v_max[0]], [v_min[0], v_max[0]], 
            lw = 2, c = 'red', zorder = 10, label = "Equal")
    plt.scatter(target[:,0], predictions[:,0], 
            label = "Testing data")
    plt.xlabel('Mole Fraction')
    plt.ylabel('Predictions')
    plt.legend()
    file_name = plot_name + "-Fv.eps"
    plt.savefig(file_name)    
    file_name = plot_name + "-Fv.pdf"
    plt.savefig(file_name)    
    plt.show()

    for i in range(nstatus - 1):
        plt.clf()
        plt.plot([v_min[i+1], v_max[i+1]], [v_min[i+1], v_max[i+1]], 
                lw = 2, c = 'red', zorder = 10, label = "Equal")
        plt.scatter(target[:,i+1], predictions[:,i+1], 
                label = "Testing data")

        name = 'K' + str(i+1)
        plt.xlabel(name)
        plt.ylabel('Predictions')
        plt.legend()
        file_name = plot_name + "-" + name + ".eps"
        plt.savefig(file_name)    
        file_name = plot_name + "-" + name + ".pdf"
        plt.savefig(file_name)    
        plt.show()

    error = predictions - target

    plt.clf()
    plt.hist(error[:,0], bins = 50)
    plt.xlabel("Mole Fraction Prediction Error")
    plt.ylabel("Count")
    file_name = plot_name + "-Fv-error.eps"
    plt.savefig(file_name)    
    file_name = plot_name + "-Fv-error.pdf"
    plt.savefig(file_name)    
    plt.show()

    for i in range(nstatus - 1):
        plt.clf()
        plt.hist(error[:,i+1], bins = 50)
        name = 'K' + str(i+1) + ' Prediction Error'
        plt.xlabel(name)
        plt.ylabel("Count")
        file_name = plot_name + '-K' + str(i+1) + "-error.eps"
        plt.savefig(file_name)    
        file_name = plot_name + '-K' + str(i+1) + "-error.pdf"
        plt.savefig(file_name)    
        plt.show()
    

def train(train_data, test_data, trans = None,
        hidden_layer = 1, hidden_cells = [10], batch_size = 30, epoch = 100, 
        validation_split = 0.1, validation_data = None,
        GL = 0.1, GL_epoch_strip = 100, min_epoch = 1000, train_number = 10, 
        plot = False, plot_name = None):

    time_begin = time.time()

    # training featrue and target
    feature_scale = []
    target_scale = []

    train = read_data(train_data)
    feature_scale_train = get_data_scale(train['feature'])
    target_scale_train = get_data_scale(train['target'])

    #print(len(feature[0]), len(target), feature_scale, target_scale)
    print "*** Number of training examples: " + str(len(train['target']))
    
    # testing featrue and target
    test = read_data(test_data)
    feature_scale_test = get_data_scale(test['feature'])
    target_scale_test = get_data_scale(test['target'])

    print "*** Number of testing examples: " + str(len(test['target']))

    feature_scale = get_data_rescale(feature_scale_train, feature_scale_test)
    target_scale = get_data_rescale(target_scale_train, target_scale_test)

    validation = None
    has_val_data = False
    if validation_data is not None:
        validation = read_data(validation_data)
        feature_scale_val = get_data_scale(validation['feature'])
        target_scale_val = get_data_scale(validation['target'])

        feature_scale = get_data_rescale(feature_scale, feature_scale_val)
        target_scale = get_data_rescale(target_scale, target_scale_val)

        scale_feature(validation['feature'], feature_scale)
        scale_target(validation['target'], target_scale, 
                method = trans)

        has_val_data = True
        print "*** Number of validation examples: " + str(len(validation['target']))

    scale_feature(train['feature'], feature_scale)
    scale_target(train['target'], target_scale, method = trans)

    scale_feature(test['feature'], feature_scale)
    scale_target(test['target'], target_scale, method = trans)

    loss_opt = 1.0;
    loss_history_opt = None
    W_opt = None
    b_opt = None
    pred_opt = None

    print "*** Training begins ..."
    for i in range(train_number):
        print("============ TRAINING PROCESS: %d ============ " %(i + 1))
        W, b, loss, pred, loss_history = train_model(
                train, test, 
                hidden_layer = hidden_layer, hidden_cells = hidden_cells, 
                batch_size = batch_size, epoch = epoch, 
                validation_split = validation_split, has_val_data = has_val_data, validation_data = validation,
                GL = GL, GL_epoch_strip = GL_epoch_strip, min_epoch = min_epoch, 
                plot = plot, plot_name = plot_name)
        if loss < loss_opt:
            loss_opt = loss
            W_opt = W
            b_opt = b
            pred_opt = pred
            loss_history_opt = loss_history

    scale_back_target(test['target'], target_scale,
            method = trans)
    scale_back_target(pred_opt, target_scale,
            method = trans)

    time_end = time.time()
    print("Time cost: %f seconds" %(time_end - time_begin))

    if (plot):
        plot_test_result(test['target'], pred_opt, plot_name)

    if (plot):
        plt.clf()
        plt.figure()
        plt.xlabel('Epoch')
        plt.yscale('log')
        plt.ylabel('Loss')
        plt.plot(loss_history_opt[0], loss_history_opt[1], label='Train Loss')
        plt.plot(loss_history_opt[0], loss_history_opt[2], label='Validation Loss')
        plt.legend()

        file_name = plot_name + "-loss.eps"
        plt.savefig(file_name)    
        file_name = plot_name + "-loss.pdf"
        plt.savefig(file_name)    

        plt.show()


    return ANN_SPLIT(W = W_opt, b = b_opt, scale = {'feature': feature_scale, 'target': target_scale})


def read_weights_from_file(file_name, level):
    input_name = None
    W = [None] * level
    b = [None] * level
    
    for i in range(level):
        input_name = file_name + '-W' + str(i) + '.csv'
        W[i] = np.loadtxt(input_name, delimiter=',')
        
        input_name = file_name + '-b' + str(i) + '.csv'
        b[i] = np.loadtxt(input_name, delimiter=',')
        
    input_name = file_name + '-scale-f.csv'
    feature_scale = np.loadtxt(input_name, delimiter=',')

    input_name = file_name + '-scale-t.csv'
    target_scale = np.loadtxt(input_name, delimiter=',')

    return W, b, {'feature': feature_scale, 'target': target_scale}

def load_model(W, b):

    level = len(W)

    nfeature = W[0].shape[0]
    nstatus = W[level - 1].shape[1]

    model = keras.Sequential()
    model.add(keras.layers.Dense(W[0].shape[1], 
                activation = tf.nn.softmax,
                input_shape = (nfeature,)))

    for i in range(level - 2):
        model.add(keras.layers.Dense(W[i+1].shape[1],
                activation = tf.nn.softmax))

    model.add(keras.layers.Dense(nstatus))

    i = 0
    for layer in model.layers:
        layer.set_weights([W[i], b[i]])
        i = i + 1

    return model

def to_file(W, b, scale, file_name = None):
    output_name = None
    
    if file_name is None:
        file_name = 'ANN-SPLIT-model-'
    
    level = len(W)
    
    for i in range(level):
        output_name = file_name + '-W' + str(i) + '.csv'
        np.savetxt(output_name, W[i], delimiter = ',')
        
        output_name = file_name + '-b' + str(i) + '.csv'
        np.savetxt(output_name, b[i].reshape([1, b[i].shape[0]]), delimiter = ',')

    output_name = file_name + '-scale-f.csv'
    np.savetxt(output_name, scale['feature'], delimiter = ',')

    output_name = file_name + '-scale-t.csv'
    np.savetxt(output_name, scale['target'], delimiter = ',')

def predict_split(model, scale, feature, trans = None):
    scale_feature(feature, scale['feature'])

    pred = model.predict(feature)

    scale_back_feature(feature, scale['feature'])
    scale_back_target(pred, scale['target'],
            method = trans)

    return pred

def compare_prediction(file_name, model, scale):
    data = read_data(file_name)

    pred = predict_split(model, scale, data['feature'])
    target = data['target']

    diff = pred - target

    return {'target': target, 'prediction': pred, 'difference': diff}














