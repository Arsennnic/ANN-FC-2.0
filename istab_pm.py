import pandas as pd
import ternary
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import time
import math

class ANN_STAB:
    """
        ANN model for stability test
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
        self.pred = None

    def store_model(self, file_name):
        to_file(self.W, self.b, self.scale, file_name)


    def stab_test(self, file_name, safeguard = 0.0):
        self.pred = stability_test(file_name, self.model, self.scale, 
                safeguard = safeguard)

        return self.pred

    def plot(self, plot_name = None):
        if self.pred is not None:
            plot_prediction_ternary(self.pred, plot_name = plot_name)
        



def read_data(data_file, rescaling = False, 
        feature_scale = None, target_scale = None,
        skip_header = 1):

    data = np.genfromtxt(data_file, delimiter=',', skip_header = skip_header)

    ncomp = data.shape[1] - 2

    feature0 = data[:, 0:ncomp]
    target_begin = ncomp
    target0 = data[:,target_begin:]

    feature = []
    target = []
    for f, t in zip(feature0, target0):
        if t[0] > t[1]:
            feature.append(f)
            target.append(t)
        else:
            print f, t

    feature = np.array(feature)
    target = np.array(target)

    if rescaling:
        nc = target.shape[1]
        nr = target.shape[0]

        for i in range(nc):
            max_v = target[:,i].max()
            min_v = target[:,i].min()

            target[:,i] = (target[:,i] - min_v) / (max_v - min_v)

            if target_scale is not None:
                target_scale.append(min_v)
                target_scale.append(max_v)

        nc = feature.shape[1]
        nr = feature.shape[0]

        for i in range(nc):
            max_v = feature[:,i].max()
            min_v = feature[:,i].min()

            feature[:,i] = (feature[:,i] - min_v) / (max_v - min_v)

            if feature_scale is not None:
                feature_scale.append(min_v)
                feature_scale.append(max_v)
        
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


def scale_feature(data, scale):
    """
        scale feature: min-max
    """
    n = data.shape[1]

    for i in range(n):
        data[:,i] = data_trans(data[:,i], 
                scale[i*2], scale[i*2+1])


def scale_target(data, scale, method = None):
    """
        target scale
    """
    n = data.shape[1]

    data[:,0] = data_trans(data[:,0], scale[0], scale[1])

    for i in range(n - 1):
        data[:,i+1] = data_trans(data[:,i+1], 
                scale[(i+1)*2], scale[(i+1)*2+1], method = method)

def scale_back_feature(data, scale):
    """
        back to target scale
    """
    n = data.shape[1]

    for i in range(n):
        data[:,i] = data_detrans(data[:,i], 
                scale[i*2], scale[i*2+1])


def scale_back_target(data, scale, method = None):
    """
        back to target scale
    """
    n = data.shape[1]

    data[:,0] = data_detrans(data[:,0], scale[0], scale[1])

    for i in range(n - 1):
        data[:,i+1] = data_detrans(data[:,i+1], 
                scale[(i+1)*2], scale[(i+1)*2+1], method = method)

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.')



def plot_history(history):
    plt.clf()
    plt.figure()
    plt.xlabel('Epoch')
    plt.yscale('log')
    plt.ylabel('Loss ')

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
                            hidden_layer = 1, hidden_cells = [10], batch_size = 30, 
                            epoch = 100, GL = 0.1, GL_epoch_strip = 100, min_epoch = 1000, 
                            validation_split = 0.1, has_val_data = False, validation_data = None,
                            plot = True, plot_name = None):

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

    filepath = './stab_weights'
    #filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
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

    model.load_weights(filepath)

    W = []
    b = []
    for layer in model.layers:
        W.append(layer.get_weights()[0])
        b.append(layer.get_weights()[1])

    [loss, mae] = model.evaluate(feature_test, target_test, verbose=0)

    print("    Training Loss: %1.5f, MAE: %1.5f" %(loss, mae))

    test_predictions = model.predict(feature_test)

    return W, b, loss, test_predictions, loss_history

def plot_test_result(target, predictions, plot_name):
    target_min = target.min(axis=0)
    target_max = target.max(axis=0)
    pred_min = predictions.min(axis=0)
    pred_max = predictions.max(axis=0)

    pu_min = np.minimum(target_min[0], pred_min[0])
    pu_max = np.maximum(target_max[0], pred_max[0])

    plt.clf()
    plt.plot([pu_min, pu_max], [pu_min, pu_max], 
            lw = 2, c = 'red', zorder = 10, label = "Equal")
    plt.scatter(target[:,0], predictions[:,0], label = "Testing data")
    plt.xlabel('True Upper Saturation Pressure')
    plt.ylabel('Predictions')
    plt.legend()
    file_name = plot_name + "-Psu.eps"
    plt.savefig(file_name, bbox_inches='tight',pad_inches=0)    
    file_name = plot_name + "-Psu.pdf"
    plt.savefig(file_name, bbox_inches='tight',pad_inches=0)    
    plt.show()

    filted_target = []
    filted_prediction = []
    for t, p in zip(target[:,1], predictions[:,1]):
        if t > 1.0:
            filted_target.append(t)
            filted_prediction.append(p)

    filted_target = np.array(filted_target)
    filted_prediction = np.array(filted_prediction)

    target_min = filted_target.min()
    target_max = filted_target.max()
    pred_min = filted_prediction.min()
    pred_max = filted_prediction.max()

    pl_min = np.minimum(target_min, pred_min)
    pl_max = np.maximum(target_max, pred_max)

    plt.clf()
    plt.plot([pl_min, pl_max], [pl_min, pl_max], 
            lw = 2, c = 'red', zorder = 10, label = "Equal")
    plt.scatter(filted_target, filted_prediction, label = "Tesing data")
    plt.xlabel('True Lower Saturation Pressure')
    plt.ylabel('Predictions')
    plt.legend()
    file_name = plot_name + "-Psl.eps"
    plt.savefig(file_name, bbox_inches='tight',pad_inches=0)    
    file_name = plot_name + "-Psl.pdf"
    plt.savefig(file_name, bbox_inches='tight',pad_inches=0)    
    plt.show()

    plt.clf()
    error_upper = predictions[:,0] - target[:,0]
    plt.hist(error_upper, bins = 50)
    plt.xlabel("Prediction Error: Upper Saturation Pressure")
    plt.ylabel("Count")
    file_name = plot_name + "-Psu-error.eps"
    plt.savefig(file_name, bbox_inches='tight',pad_inches=0)    
    file_name = plot_name + "-Psu-error.pdf"
    plt.savefig(file_name, bbox_inches='tight',pad_inches=0)    
    plt.show()

    plt.clf()
    error_lower = filted_prediction - filted_target
    plt.hist(error_lower, bins = 50)
    plt.xlabel("Prediction Error: Lower Saturation Pressure")
    plt.ylabel("Count")
    file_name = plot_name + "-Psl-error.eps"
    plt.savefig(file_name, bbox_inches='tight',pad_inches=0)    
    file_name = plot_name + "-Psl-error.pdf"
    plt.savefig(file_name, bbox_inches='tight',pad_inches=0) 
    plt.show()


def train(train_data, test_data, trans = None,
        hidden_layer = 1, hidden_cells = [10], 
        batch_size = 30, epoch = 100, 
        validation_split = 0.1, validation_data = None,
        GL = 0.1, GL_epoch_strip = 100, min_epoch = 1000, train_number = 10, 
        plot = True, plot_name = None):

    time_begin = time.time()

    # training featrue and target
    feature_scale = []
    target_scale = []
    train = read_data(train_data, rescaling = True, 
            feature_scale = feature_scale,
            target_scale = target_scale)
    print "*** Number of training examples: " + str(len(train['target']))
    
    # testing featrue and target
    test = read_data(test_data)
    scale_feature(test['feature'], feature_scale)
    scale_target(test['target'], target_scale, method = trans)
    print "*** Number of testing examples: " + str(len(test['target']))

    validation = None
    has_val_data = False
    if validation_data is not None:
        validation = read_data(validation_data)

        scale_feature(validation['feature'], feature_scale)
        scale_target(validation['target'], target_scale, 
                method = trans)
        has_val_data = True
        print "*** Number of validation examples: " + str(len(validation['target']))


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
        plt.savefig(file_name, bbox_inches='tight',pad_inches=0)    
        file_name = plot_name + "-loss.pdf"
        plt.savefig(file_name, bbox_inches='tight',pad_inches=0)    

    return ANN_STAB(W = W_opt, b = b_opt, scale = {'feature': feature_scale, 'target': target_scale})



def to_file(W, b, scale, file_name = None):
    output_name = None
    
    if file_name is None:
        file_name = 'ANN-STAB-PM-model-'
    
    level = len(W)
    
    for i in range(level):
        output_name = file_name + '-W' + str(i) + '.csv'
        np.savetxt(output_name, W[i], delimiter = ',')
        
        output_name = file_name + '-b' + str(i) + '.csv'
        np.savetxt(output_name, b[i].reshape([1,b[i].shape[0]]), delimiter = ',')

    output_name = file_name + '-scale-f.csv'
    np.savetxt(output_name, scale['feature'], delimiter = ',')

    output_name = file_name + '-scale-t.csv'
    np.savetxt(output_name, scale['target'], delimiter = ',')

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


def predict_ps(model, scale, feature, trans = None):
    scale_feature(feature, scale['feature'])

    pred = model.predict(feature)

    scale_back_feature(feature, scale['feature'])
    scale_back_target(pred, scale['target'],
            method = trans)

    return pred

def predict_stability(model, scale, z, p, 
        safeguard = 0.0,
        trans = None):
    ps = predict_ps(model, scale, z, trans)

    pu = ps[:,0]
    pl = ps[:,1]

    unknown = ((np.abs(pu - p) < safeguard) | (np.abs(pl - p) < safeguard)) & (pl < pu) 
    unstable = (~unknown) & (pu > pl) & (p > pl) & (p < pu) & (pu > 1.0)
    stable = ~(unknown | unstable)

    return {'stable': stable, 'unstable': unstable, 'unknown': unknown}

def prediction_result(z, p, pred_stab, real_stab):
    stable = pred_stab['stable']
    unstable = pred_stab['unstable']
    unknown = pred_stab['unknown']

    correct = (stable & real_stab) | (unstable & ~real_stab)
    wrong = ~unknown & ~correct

    z_correct = z[correct]
    p_correct = p[correct]

    z_wrong = z[wrong]
    p_wrong = p[wrong]

    z_unknown = z[unknown]
    p_unknown = p[unknown]

    return {'correct': [z_correct, p_correct], 'wrong': [z_wrong, p_wrong], 'unknown': [z_unknown, p_unknown]} 

def read_stab_data(file_name):

    data = np.genfromtxt(file_name, delimiter=',', 
            skip_header = False)

    nc = data.shape[1]
    nr = data.shape[0]

    # the rest are the composition of each component
    ncomp = nc - 2
    
    # feature (composition)
    z = data[:, 0:ncomp]

    # presure
    p = data[:, ncomp:ncomp+1]
    p = p.reshape(p.shape[0])

    # target (unstable: 0, stable: 1)
    stab = data[:, (ncomp + 1):(ncomp + 2)] > 0.5
    stab = stab.reshape(stab.shape[0])
    
    return z, p, stab 


def stability_test(file_name, model, scale, safeguard = 0.0):

    z, p, real_stab = read_stab_data(file_name)

    pred_stab = predict_stability(model, scale, 
            z, p, safeguard = safeguard)

    result = prediction_result(z, p, pred_stab, real_stab)

    print("Correct: %d" %(len(result['correct'][0])))
    print("Wrong: %d" %(len(result['wrong'][0])))
    print("Uncertain: %d" %(len(result['unknown'][0])))
    print("Accuracy: %e" %(1.0 - float(len(result['wrong'][0])) / float(len(real_stab))))

    return result

def plot_prediction_ternary(result, scale = 100, multiple = 20, 
        plot_name = None):

    correct = result['correct']
    wrong = result['wrong']
    unknown = result['unknown']

    fontsize = 12
    axis_fontsize = 8

    figure, tax = ternary.figure(scale = scale)
    #tax.set_title("Training Data", fontsize=20)
    tax.boundary(linewidth=1.0)
    tax.gridlines(multiple = multiple, color="blue")

    tax.left_axis_label("$C_{10}$", fontsize=fontsize)
    tax.right_axis_label("$C_{6}$", fontsize=fontsize)
    tax.bottom_axis_label("$C_{1}$", fontsize=fontsize)

    correct_points = correct[0] * scale
    wrong_points = wrong[0] * scale
    unknown_points = unknown[0] * scale

    correct_points = np.rint(correct_points)
    correct_points = correct_points.astype(int)

    wrong_points = np.rint(wrong_points)
    wrong_points = wrong_points.astype(int)

    unknown_points = np.rint(unknown_points)
    unknown_points = unknown_points.astype(int)

    if (correct_points.shape[0] > 0):
        tax.scatter(correct_points, s = 1, color='red', label="Correct")
    if (wrong_points.shape[0] > 0):
        tax.scatter(wrong_points, s = 1, color='blue', label="Wrong")
    if (unknown_points.shape[0] > 0):
        tax.scatter(unknown_points, s = 1, color='green', label="Uncertain")

    tax.legend()
    tax.clear_matplotlib_ticks()
    tax.ticks(axis='lbr', linewidth=1, multiple=multiple, fontsize = 8)
    tax._redraw_labels()
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    if plot_name is None:
        plot_name = 'stab-prediction-result'

    file_name = plot_name + '.eps'
    tax.savefig(file_name)
    file_name = plot_name + '.pdf'
    tax.savefig(file_name)

    

