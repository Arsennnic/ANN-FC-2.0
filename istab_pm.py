import pandas as pd
import ternary
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import time
import math

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

def NN_STAB_scale_matrix_bias(W, b, feature_scale, target_scale):
    P_min = target_scale[0]
    P_max = target_scale[1]

    W[-1][:,0] = (P_max - P_min) * W[-1][:,0]
    b[-1][0] = (P_max - P_min) * b[-1][0] + P_min

    P_min = target_scale[2]
    P_max = target_scale[3]

    W[-1][:,1] = (P_max - P_min) * W[-1][:,1]
    b[-1][1] = (P_max - P_min) * b[-1][1] + P_min
    
    nf = W[0].shape[0]

    B = np.zeros(nf)
    for i in range(nf):
        v_min = feature_scale[i*2]
        v_max = feature_scale[i*2+1]

        if (math.fabs(v_max - v_min) > 1e-5):
            B[i] = v_min / (v_max - v_min)

    b[0] = b[0] - np.matmul(B, W[0])
 
    M = np.eye(nf)
    for i in range(nf):
        v_min = feature_scale[i*2]
        v_max = feature_scale[i*2+1]

        if math.fabs(v_max - v_min) > 1e-5:
            M[i][i] = 1.0 / (v_max - v_min)

    W[0] = np.matmul(M, W[0])

def NN_STAB_predict_saturation_pressure(W, b, feature = None, target = None, 
                                        data_file = None, plot = True, plot_name = None,
                                        output = True, output_file = None):
    size = len(W)
    
    if feature is None:
        if data_file is None:
            return
        else:
            feature, target = read_data(data_file, 
                    for_training = False)
    
    #print(target[0:100])
    nfeature = len(feature[0])
    
    y = [None] * size
    
    x = tf.placeholder(tf.float32, [None, nfeature])
    
    input_x = x
    for i in range(size - 1):
        y[i] = tf.nn.softmax(tf.matmul(input_x, W[i].astype(np.float32)) + b[i], name="output")
        input_x = y[i]
            
    y[size - 1] = tf.matmul(input_x, W[size - 1].astype(np.float32)) + b[size - 1]
    
    if target is not None:
        max_diff = tf.norm(y[size - 1] - target)
        max_index = tf.argmax(y[size - 1] - target)
    else:
        max_diff = tf.zeros([1])
        max_index = 0
        
    pred = None
    L_inf = None
    with tf.Session() as sess:
        [pred, L_inf, L_index] = sess.run([y[size - 1], max_diff, max_index], feed_dict={x: feature})
        
    print("L inf = %f" %L_inf)
    print L_index
    print feature[L_index], target[L_index], pred[L_index]

    if output:
        if output_file is None:
            output_file = "ps-prediction.csv"

        output_value = []
        for f, p in zip(feature, pred):
            value = []

            for f0 in f:
                value.append(f0)
            for p0 in p:
                value.append(p0)

            output_value.append(value)

        np.savetxt(output_file, output_value, delimiter=",")

    if plot:
        composition = []
        for f in feature:
            composition.append(f[0])
        
        plt.clf()
        plt.xlabel("Composition")
        plt.ylabel("Pressure, atm")
        
        plt.plot(composition, pred, label='ANN-STAB model', color = 'red')
        if target is not None:
            #print(target[0:100])
            plt.plot(composition, target, label='Real Saturation', color = 'blue')
        plt.legend(bbox_to_anchor = (0.05, 1.0), loc = "upper left")
        plt.show()
        
        file_name = None
        
        if plot_name is None:
            plot_name = "ANN-STAB-PM-saturation-envelope"
        
        file_name = plot_name + ".eps"    
        plt.savefig(file_name)
        
        file_name = plot_name + ".pdf"
        plt.savefig(file_name)
    
    return feature, target, pred, L_inf


def NN_STAB_data_transformation(data, data_min, data_max, 
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



def NN_STAB_data_detransformation(data, data_min, data_max,
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


def NN_STAB_scale_feature(data, scale):
    """
        scale feature: min-max
    """
    n = data.shape[1]

    for i in range(n):
        data[:,i] = NN_STAB_data_transformation(data[:,i], 
                scale[i*2], scale[i*2+1])


def NN_STAB_scale_target(data, scale, method = None):
    """
        target scale
    """
    n = data.shape[1]

    data[:,0] = NN_STAB_data_transformation(data[:,0], scale[0], scale[1])

    for i in range(n - 1):
        data[:,i+1] = NN_STAB_data_transformation(data[:,i+1], 
                scale[(i+1)*2], scale[(i+1)*2+1], method = method)

def NN_STAB_scale_back_target(data, scale, method = None):
    """
        back to target scale
    """
    n = data.shape[1]

    data[:,0] = NN_STAB_data_detransformation(data[:,0], scale[0], scale[1])

    for i in range(n - 1):
        data[:,i+1] = NN_STAB_data_detransformation(data[:,i+1], 
                scale[(i+1)*2], scale[(i+1)*2+1], method = method)

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.')



def plot_history(history):
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
            self.epoch_opt = epoch

        if (epoch + 1) % 100 == 0:
            self.min_val_loss_batch = 1.0

        if self.min_val_loss_batch > val_loss:
            self.min_val_loss_batch = val_loss

        if (epoch + 1) % 100 == 0:
            self.GL = self.min_val_loss_batch / self.min_val_loss - 1.0
            print("    Epoch %05d: MGL: (%1.4f / %1.4f) - 1.0 = %1.4f" 
                    % (epoch + 1, self.min_val_loss_batch, self.min_val_loss, self.GL))

        if (epoch > self.min_epoch and (epoch + 1) % 100 == 0):
            if self.GL > self.alpha:
                print("    Earlystopping! Best Performance epoch %d" %(self.epoch_opt))
                self.model.stop_training = True

def NN_train_phase_envelope(train_data, test_data, 
                            hidden_layer = 1, hidden_cells = [10], batch_size = 30, 
                            epoch = 100, GL = 0.1, min_epoch = 1000, 
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

    earlystop = EarlyStoppingByGL(alpha = GL, min_epoch = min_epoch)

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
             
    plot_history(history)

    model.load_weights(filepath)

    W = []
    b = []
    for layer in model.layers:
        W.append(layer.get_weights()[0])
        b.append(layer.get_weights()[1])

    [loss, mae] = model.evaluate(feature_test, target_test, verbose=0)

    print("    Training Loss: %1.5f, MAE: %1.5f" %(loss, mae))

    test_predictions = model.predict(feature_test)

    return W, b, loss, test_predictions

def plot_test_result(target, predictions, plot_name):
    target_min = target.min(axis=0)
    target_max = target.max(axis=0)
    pred_min = predictions.min(axis=0)
    pred_max = predictions.max(axis=0)

    pu_min = np.minimum(target_min[0], pred_min[0])
    pu_max = np.maximum(target_max[0], pred_max[0])

    plt.plot([pu_min, pu_max], [pu_min, pu_max], 
            lw = 2, c = 'red', zorder = 10, label = "Equal")
    plt.scatter(target[:,0], predictions[:,0], label = "Testing data")
    plt.xlabel('True Upper Saturation Pressure')
    plt.ylabel('Predictions')
    plt.legend()
    plt.show()
    file_name = plot_name + "-Psu.eps"
    plt.savefig(file_name)    
    file_name = plot_name + "-Psu.pdf"
    plt.savefig(file_name)    

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

    plt.plot([pl_min, pl_max], [pl_min, pl_max], 
            lw = 2, c = 'red', zorder = 10, label = "Equal")
    plt.scatter(filted_target, filted_prediction, label = "Tesing data")
    plt.xlabel('True Lower Saturation Pressure')
    plt.ylabel('Predictions')
    plt.legend()
    plt.show()
    file_name = plot_name + "-Psl.eps"
    plt.savefig(file_name)    
    file_name = plot_name + "-Psl.pdf"
    plt.savefig(file_name)    

    error_upper = predictions[:,0] - target[:,0]
    plt.hist(error_upper, bins = 50)
    plt.xlabel("Prediction Error: Upper Saturation Pressure")
    plt.ylabel("Count")
    plt.show()
    file_name = plot_name + "-Psu-error.eps"
    plt.savefig(file_name)    
    file_name = plot_name + "-Psu-error.pdf"
    plt.savefig(file_name)    

    error_lower = filted_prediction - filted_target
    plt.hist(error_lower, bins = 50)
    plt.xlabel("Prediction Error: Lower Saturation Pressure")
    plt.ylabel("Count")
    plt.show()
    file_name = plot_name + "-Psl-error.eps"
    plt.savefig(file_name)    
    file_name = plot_name + "-Psl-error.pdf"
    plt.savefig(file_name)    


def train(train_data, test_data, trans = None,
        hidden_layer = 1, hidden_cells = [10], 
        batch_size = 30, epoch = 100, 
        validation_split = 0.1, validation_data = None,
        GL = 0.1, min_epoch = 1000, train_number = 10, 
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
    NN_STAB_scale_feature(test['feature'], feature_scale)
    NN_STAB_scale_target(test['target'], target_scale, method = trans)
    print "*** Number of testing examples: " + str(len(test['target']))

    validation = None
    has_val_data = False
    if validation_data is not None:
        validation = read_data(validation_data)

        NN_STAB_scale_feature(validation['feature'], feature_scale)
        NN_STAB_scale_target(validation['target'], target_scale, 
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
        W, b, loss, pred = NN_train_phase_envelope(
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

    NN_STAB_scale_back_target(test['target'], target_scale,
            method = trans)
    NN_STAB_scale_back_target(pred_opt, target_scale,
            method = trans)

    time_end = time.time()
    print("Time cost: %f seconds" %(time_end - time_begin))

    if (plot):
        plot_test_result(test['target'], pred_opt, plot_name)

    return W_opt, b_opt, loss_opt, {'feature': feature_scale, 'target': target_scale}



def to_file(W, b, scale, file_name = None):
    output_name = None
    
    if file_name is None:
        file_name = 'ANN-STAB-PM-model-'
    
    level = len(W)
    
    for i in range(level):
        output_name = file_name + 'W' + str(i) + '.csv'
        np.savetxt(output_name, W[i], delimiter = ',')
        
        output_name = file_name + 'b' + str(i) + '.csv'
        np.savetxt(output_name, b[i], delimiter = ',')

    output_name = file_name + 'scale-f.csv'
    np.savetxt(output_name, scale['feature'], delimiter = ',')

    output_name = file_name + 'scale-t.csv'
    np.savetxt(output_name, scale['target'], delimiter = ',')


def load_model(file_name, level):
    input_name = None
    W = [None] * level
    b = [None] * level
    
    for i in range(level):
        input_name = file_name + 'W' + str(i) + '.csv'
        W[i] = np.loadtxt(input_name, delimiter=',')
        
        input_name = file_name + 'b' + str(i) + '.csv'
        b[i] = np.loadtxt(input_name, delimiter=',')
        
    input_name = file_name + 'scale-f.csv'
    feature_scale = np.loadtxt(input_name, delimiter=',')

    input_name = file_name + 'scale-t.csv'
    target_scale = np.loadtxt(input_name, delimiter=',')

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

    return model, {'feature': feature_scale, 'target': target_scale}

def predict_ps(model, scale, feature, trans = None):
    NN_STAB_scale_feature(feature, scale['feature'])

    pred = model.predict(feature)

    NN_STAB_scale_back_target(pred, scale['target'],
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
    wrong = ~unknow & ~correct

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

    # 3 values for "liquid", "vapor", and "unstable"
    # 1 value for pressure
    # the rest are the composition of each component
    ncomp = nc - 4
    
    # feature (composition)
    z = data[:, 0:ncomp]

    # presure
    p = data[:, ncomp:ncomp+1]

    # target (unstable: 0, stable: 1)
    stab = ~(data[:, (ncomp + 1):(ncomp + 2)] > 0.5)
    
    return z, p, stab 


def stability_test(file_name, model, scale):

    z, p, real_stab = read_stab_data(file_name)

    pred_stab = predict_stability(model, scale, z, p, safeguard = 0.0)

    result = prediction_result(z, p, pred_stab, real_stab)

    print("Correct: %d" %(len(result['correct'][0])))
    print("Wrong: %d" %(len(result['wrong'][0])))
    print("Uncertain: %d" %(len(result['unknown'][0])))
    print("Accuracy: %e" %(1.0 - float(len(result['wrong'][0])) / float(len(real_stab))))

    return result

def plot_prediction_ternary(result, scale = 100, multiple = 5, 
        plot_name = None):

    correct = result['correct']
    wrong = result['wrong']
    unknown = result['unknown']

    fontsize = 12
    axis_fontsize = 8

    figure, tax = ternary.figure(scale = scale)
    #tax.set_title("Training Data", fontsize=20)
    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple = multiple, color="blue")

    tax.left_axis_label("$C_{10}$", fontsize=fontsize)
    tax.right_axis_label("$C_{6}$", fontsize=fontsize)
    tax.bottom_axis_label("$C_{1}$", fontsize=fontsize)

    correct_points = correct[0]
    wrong_points = wrong[0]
    unknown_points = unknown[0]

    tax.scatter(correct_points, color='red', label="Correct")
    tax.scatter(wrong_points, color='blue', label="Wrong")
    tax.scatter(unknown_points, color='green', label="Uncertain")

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

    

def predict(W, b, data_file, delta_p = 0.0, safeguard = 1.0, 
        plot = True, plot_name = None):

    time_begin = time.time()
    stable, wrong_prediction, unknown_prediction, result = NN_STAB_predict_stability(W, b, 
            data_file, delta_p = delta_p, safeguard = safeguard, 
            plot = plot, plot_name = plot_name)
    time_end = time.time()

    print("Prediction time cost: %f seconds" %(time_end - time_begin))

    return stable, wrong_prediction, unknown_prediction, result


def plot_wrong_prediction(data_file_i, data_file_d, 
        unknown_prediction, wrong_prediction, 
        plot = True, plot_name = None):

    feature_i, target_i = read_data(data_file_i)
    feature_d, target_d = read_data(data_file_d)

    testing_data_C = []
    testing_data_P = []

    for f, t in zip(feature_i, target_i):
        testing_data_C.append(f[0])
        testing_data_P.append(t[0])
    
    for f, t in zip(feature_d, target_d):
        testing_data_C.append(f[0])
        testing_data_P.append(t[0])

    C_min = 0.5 * (np.amin(testing_data_C) + 0.0)
    C_max = 0.5 * (np.amax(testing_data_C) + 1.0)
    P_min = np.amin(testing_data_P)
    P_max = np.amax(testing_data_P)
    
    plt.xlabel("Composition")
    plt.ylabel("Pressure, atm")
    plt.xlim((C_min, C_max))
    plt.ylim((P_min, P_max))
    plt.plot(testing_data_T, testing_data_P, 
             label = 'Real saturation pressure envelope', c = 'red')
    plt.scatter(unknown_prediction[0], unknown_prediction[1], 
                label = 'No prediction points', c = 'blue', s = 5.0)
    plt.scatter(wrong_prediction[0], wrong_prediction[1], 
                label = 'Wrong prediction points', c = 'green', s = 15.0)
    plt.legend(bbox_to_anchor = (0.05, 1.0), loc = "upper left")

    output_name = None
    
    if plot_name is None:
        plot_name = "ANN-STAB-prediction-wrong-unknown-points"
        
    output_name = plot_name + ".eps"
    plt.savefig(output_name)
    
    output_name = plot_name + ".pdf"
    plt.savefig(output_name)

