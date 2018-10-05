import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import time
import math

def NN_STAB_data_generation(data_file, rescaling = False, 
        feature_scale = None, target_scale = None,
        skip_header = 0, for_training = False):

    data = np.genfromtxt(data_file, delimiter=',', skip_header = skip_header)

    ncomp = data.shape[1] - 2

    feature0 = data[:, 0:ncomp]
    target_begin = ncomp
    target0 = data[:,target_begin:]

    feature = []
    target = []
    for f, t in zip(feature0, target0):
        if for_training:
            if t[0] > t[1]:
                feature.append(f)
                target.append(t)
        else:
            feature.append(f)
            target.append(t)

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
        
    return feature, target

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
            feature, target = NN_STAB_data_generation(data_file, 
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


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.')



def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
            label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
            label = 'Val loss')
    plt.legend()
    plt.show()

def NN_train_phase_envelope(train_data, test_data, 
                            hidden_cells = 10, batch_size = 30, 
                            epoch = 100, plot = True, plot_name = None):
    time_begin = time.time()
    # training featrue and target
    print("Read training data:")
    feature_scale = []
    target_scale = []
    feature, target = NN_STAB_data_generation(train_data, 
            rescaling = True, feature_scale = feature_scale,
            target_scale = target_scale, for_training = True)
    print(len(feature[0]), len(target), feature_scale, target_scale)
    print("Done")
    
    # testing featrue and target
    print("Read testing data:")
    feature_test, target_test = NN_STAB_data_generation(test_data, 
            for_training = True)
    print(len(feature_test[0]), len(target_test))
    print("Done")
    
    nfeature = len(feature[0])
    nstatus = len(target[0])
    print(nfeature, nstatus)

    model = keras.Sequential([
            keras.layers.Dense(hidden_cells, 
                activation = tf.nn.softmax,
                input_shape = (nfeature,)),
            keras.layers.Dense(nstatus)
            ])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse', optimizer = optimizer, metrics=['mae'])

    history = model.fit(feature, target, epochs = epoch,
            validation_split=0.1, verbose=0)
             
    plot_history(history)

    W = []
    b = []
    for layer in model.layers:
        W.append(layer.get_weights()[0])
        b.append(layer.get_weights()[1])

    NN_STAB_scale_matrix_bias(W, b, feature_scale, target_scale)
    
    i = 0
    for layer in model.layers:
        layer.set_weights([W[i], b[i]])
        i += 1
    #pred = NN_STAB_predict_saturation_pressure(min_W, min_b, data_file = test_data, 
    #                                                       plot = plot, plot_name = plot_name)
    #print("L_inf error: %f" %(Linf_error))

    [loss, mae] = model.evaluate(feature_test, target_test, verbose=0)

    print("Testing set Mean Abs Error: ${:1.6f}".format(mae))

    test_predictions = model.predict(feature_test)
    print(test_predictions.shape)

    plt.scatter(target_test[:,0], test_predictions[:,0])
    plt.xlabel('True Upper Saturation Pressure')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    plt.plot([-100, 100], [-100, 100])
    plt.show()

    plt.scatter(target_test[:,1], test_predictions[:,1])
    plt.xlabel('True Lower Saturation Pressure')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    plt.plot([-100, 100], [-100, 100])
    plt.show()

    error = test_predictions - target_test
    plt.hist(error, bins = 50)
    plt.xlabel("Prediction Error")
    plt.ylabel("Count")
    plt.show()

    
    #return Wf, bf, min_W, min_b, pred

def train(train_data, test_data, 
                            hidden_cells = 10, batch_size = 30, 
                            epoch = 100, plot = True, plot_name = None):

    #Wf, bf, min_W, min_b, pred = NN_train_phase_envelope(train_data, test_data, 
    #                        hidden_cells = hidden_cells, 
    #                        batch_size = batch_size, epoch = epoch, 
    #                        plot = plot, plot_name = plot_name)
    NN_train_phase_envelope(train_data, test_data, 
                            hidden_cells = hidden_cells, 
                            batch_size = batch_size, epoch = epoch, 
                            plot = plot, plot_name = plot_name)

    #return Wf, bf, min_W, min_b, pred




def NN_STAB_write_matrix_bias_to_file(W, b, file_name = None):
    output_name = None
    
    if file_name is None:
        file_name = 'ANN-STAB-PM-model-'
    
    level = len(W)
    
    for i in range(level):
        output_name = file_name + 'W' + str(i) + '.csv'
        np.savetxt(output_name, W[i], delimiter = ',')
        
        output_name = file_name + 'b' + str(i) + '.csv'
        np.savetxt(output_name, b[i], delimiter = ',')


def write(W, b, file_name = None):
    NN_STAB_write_matrix_bias_to_file(W, b, file_name = file_name)


def NN_STAB_ajoint_prediction(pred_i, pred_d, 
                              plot_model_real = [True, True],
                              plot_set = None, plot_name = None):    
    ncomp = len(pred_i[0][0]) - 1
    
    # collect predicted upper saturation pressure
    feature = pred_i[0]
    pred = pred_i[2] 
    composition_i = []
    pred_pressure_i = []
    count = -1
    last_value =1.0e10
    last_X = [1.1] * ncomp
    X = [None] * ncomp
    
    for f, p in zip(feature, pred): 
        flag = False
        for i in range(ncomp):
            X[i] = f[i]
            
        if X[0] < last_X[0]:
            flag = True
                
        if flag:
            composition_i.append([])
            pred_pressure_i.append([])
            count += 1
        
        composition_i[count].append(f[0])
        pred_pressure_i[count].append(p)
        last_value = f[0]
        
        for i in range(ncomp):
            last_X[i] = X[i]
            
    Nset = len(composition_i)
    
    # collect predicted down saturation pressure
    feature = pred_d[0]
    pred = pred_d[2] 
    composition_d = []
    pred_pressure_d = []
    count = -1
    last_value = 0.0
    last_X = [0.0] * ncomp
    X = [None] * ncomp
    
    for f, p in zip(feature, pred):   
        flag = False
        for i in range(ncomp):
            X[i] = f[i]

        if X[0] > last_X[0]:
            flag = True
                
        #if flag:
        #    print(X)
                
        if flag:
            composition_d.append([])
            pred_pressure_d.append([])
            count += 1
        
        composition_d[count].append(f[0])
        pred_pressure_d[count].append(p)
        last_value = f[0]
        
        for i in range(ncomp):
            last_X[i] = X[i]
    
    
    # ajoint the composition
    composition = []
    count = 0
    for t_i, t_d in zip(composition_i, composition_d):
        composition.append([])
        
        for i in range(len(t_i)):
            composition[count].append(t_i[i])
        for i in range(len(t_d)):
            composition[count].append(t_d[i])
        
        count += 1
    

    # ajoint the predicted pressure
    pred_pressure = []
    count = 0
    for p_i, p_d in zip(pred_pressure_i, pred_pressure_d):
        pred_pressure.append([])
        
        for i in range(len(p_i)):
            pred_pressure[count].append(p_i[i])
        for i in range(len(p_d)):
            pred_pressure[count].append(p_d[i])
        
        count += 1
    
    
    real_pressure_i = None
    target = pred_i[1]
    if target is not None:
        real_pressure_i = []
        count = -1
        count_p = 0
        np = -2
        
        for p in target:
            if count_p >= np:
                real_pressure_i.append([])
                count += 1
                
                if count < len(composition_i):
                    np = len(composition_i[count])
                    
                count_p = 0
            
            real_pressure_i[count].append(p)
            count_p += 1
   
    real_pressure_d = None
    target = pred_d[1]
    if target is not None:
        real_pressure_d = []
        count = -1
        count_p = 0
        np = -2
        
        for p in target:
            if count_p >= np:
                real_pressure_d.append([])
                count += 1
                if count < len(composition_d):
                    np = len(composition_d[count])
                count_p = 0
                
            real_pressure_d[count].append(p)
            count_p += 1
    
    real_pressure = None
    if real_pressure_i is not None and real_pressure_d is not None:
        real_pressure = []
        count = 0
        for p_i, p_d in zip(real_pressure_i, real_pressure_d):
            real_pressure.append([])
        
            for i in range(len(p_i)):
                real_pressure[count].append(p_i[i])
            for i in range(len(p_d)):
                real_pressure[count].append(p_d[i])
        
            count += 1
    
    print(len(composition), len(pred_pressure), len(real_pressure))

    print(composition[0], pred_pressure[0], real_pressure[0])
    plt.clf()
    plt.xlabel("Composition")
    plt.ylabel("Pressure, atm")
    
    plot_sets = []
    if plot_set is None:
        for i in range(Nset):
            plot_sets.append(i)
    else:
        plot_sets = plot_set
        
    if plot_model_real[0]:        
        for i, set_no in enumerate(plot_sets):
            if i == 0:
                plt.plot(composition[set_no], pred_pressure[set_no],
                         label = 'ANN-STAB Model', color = 'red')
            else:
                plt.plot(composition[set_no], pred_pressure[set_no], color = 'red')
    
    #print(composition)
    #print(real_pressure)
    if real_pressure is not None:
        if plot_model_real[1]:
            for i, set_no in enumerate(plot_sets):
                if i == 0:
                    plt.plot(composition[set_no], real_pressure[set_no], 
                             label = 'Real Saturation Pressure', color = 'blue')
                else:
                    plt.plot(composition[set_no], real_pressure[set_no], color = 'blue')
    
    plt.legend(bbox_to_anchor = (0.05, 1.0), loc = "upper left")
        
    file_name = None
        
    if plot_name is None:
        plot_name = "ANN-STAB-PM-combined"
        
    file_name = plot_name + ".eps"    
    plt.savefig(file_name)
        
    file_name = plot_name + ".pdf"
    plt.savefig(file_name)



def NN_STAB_load_matrix_bias_from_file(file_name, level):
    input_name = None
    W = [None] * level
    b = [None] * level
    
    for i in range(level):
        input_name = file_name + 'W' + str(i) + '.csv'
        W[i] = np.loadtxt(input_name, delimiter=',')
        
        input_name = file_name + 'b' + str(i) + '.csv'
        b[i] = np.loadtxt(input_name, delimiter=',')
        
        #W[i] = W[i].reshape(dim)
        ##print(b[i].shape, dim)
        #print(W[i])
    dim = [W[level-1].shape[0], 1]
    W[level-1] = W[level-1].reshape(dim)

    return W, b


def read(file_name, level):
    NN_STAB_load_matrix_bias_from_file(file_name, level)



def NN_STAB_predict_stability_feature_target_from_data_file(file_name):

    data = np.genfromtxt(file_name, delimiter=',', 
            skip_header = False)

    nc = data.shape[1]
    nr = data.shape[0]

    # 3 values for "liquid", "vapor", and "unstable"
    # 1 value for pressure
    # the rest are the composition of each component
    ncomp = nc - 4
    
    # feature (composition)
    feature = data[:, 0:ncomp]

    # presure
    pressure = data[:, ncomp:ncomp+1]

    # composition
    composition = data[:, 0:1]

    # target (unstable: 0, stable: 1)
    target = ~(data[:, (ncomp + 1):(ncomp + 2)] > 0.5)
    target = target * 1.0
    
    return feature, composition, pressure, target



def NN_STAB_predict_stability(W, b, data_file, 
                              delta_p = 0.0, safeguard = 1.0, 
                              plot = True, plot_name = None):
   
    # Read feature (composition, T), pressure, and target (unstable or stable) from data file
    feature, composition, pressure, target = NN_STAB_predict_stability_feature_target_from_data_file(data_file)
    
    # Calculate upper saturation pressure and down saturation pressure
    _, _, pred_pressure, _ = NN_STAB_predict_saturation_pressure(W, b, 
                                                                feature = feature, 
                                                                plot = False)
    pred_pressure_u = pred_pressure[:,0]
    pred_pressure_d = pred_pressure[:,1]

    unknown = ((np.abs(pred_pressure_u - pressure) < safeguard * delta_p) | (np.abs(pred_pressure_d - pressure) < safeguard * delta_p)) & (pred_pressure_d < pred_pressure_u) 
        
    unstable = (~unknown) & (pred_pressure_u > pred_pressure_d) & (pressure > pred_pressure_d) & (pressure < pred_pressure_u) & (pred_pressure_u > 1.0)
    
            
    stable = ~(unknown | unstable)

    unknown = unknown * 1.0
    unstable = unstable * 1.0
    stable = stable * 1.0
    
    N = len(composition)
    C_correct = []
    C_wrong = []
    C_unknown = []
    P_correct = []
    P_wrong = []
    P_unknown = []
    Cu_ary = []
    Cs_ary = []
    Pu_ary = []
    Ps_ary = []

    for i in range(N):
        if unknown[i][0] == 1.0:
            C_unknown.append(composition[i])
            P_unknown.append(pressure[i])
        elif target[i] == stable[i][0]:
            C_correct.append(composition[i])
            P_correct.append(pressure[i])
        else:
            C_wrong.append(composition[i])
            P_wrong.append(pressure[i])
            
    wrong_prediction = [C_wrong, P_wrong]
    unknown_prediction = [C_unknown, P_unknown]
    
    if plot:
        plt.clf()
        plt.xlabel("Composition")
        plt.ylabel("Pressure, atm")
        plt.scatter(C_correct, P_correct, label= 'correct prediction', c = 'red', s = 1.0)
        plt.scatter(C_unknown, P_unknown, label = 'no prediction', c = 'blue', s = 10.0)
        plt.scatter(C_wrong, P_wrong, label = 'wrong prediction', c = 'green', s = 10.0)
        
        plt.legend(bbox_to_anchor = (0.05, 1.0), loc = "upper left")
        
        output_name = None
        if plot_name is None:
            plot_name = "ANN-STAB-PM-prediction-uniform-distributed-set"
        
        output_name = plot_name + ".eps"
        plt.savefig(output_name)
        output_name = plot_name + ".pdf" 
        plt.savefig(output_name)
        plt.show()
    
    result = []
    result.append(len(C_unknown))
    result.append(len(C_wrong))
    
    print("No prediction percentage: %f" %(float(len(C_unknown)) / float(len(target))))
    print("Wrong prediciton: %d" %(len(C_wrong)))
    print("Accuracy: %f" %(1.0 - float(len(C_wrong)) / float(len(target))))
    
    return stable, wrong_prediction, unknown_prediction, result


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

    feature_i, target_i = NN_STAB_data_generation(data_file_i)
    feature_d, target_d = NN_STAB_data_generation(data_file_d)

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

