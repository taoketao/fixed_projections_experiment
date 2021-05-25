#  See for original code (incl. code for saving):
#  /Users/morganbryant/Desktop/Files/21/01_January/passio/code-challenge/...
print('\t <><> Beginning Script <><>')
import tensorflow as tf
import os, sys
import numpy as np
from matplotlib import pyplot as plt

BATCHSIZE=1 
OLD_TEST, SANITY = False, True    # What to do with the constructed model
PRINT_WEIGHTS, PLOT_WEIGHTS_HISTOGRAM = True, True

data = tf.keras.datasets.boston_housing.load_data(test_split=0.2, seed=113)
INPUT_SHAPE = (13,)
def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)
x_train, y_train, x_test, y_test = \
        [tf.convert_to_tensor(d/np.mean(d)) for d in \
        (data[0][0], data[0][1], data[1][0], data[1][1])]

#print (data[0][0].shape, data[0][1].shape, data[1][0].shape, data[1][1].shape, )

#------ build models ------
def make_model1():
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    layer1 = tf.keras.layers.Dense( 64, activation='sigmoid')(inputs)
    layer2f1 = tf.keras.layers.Dense(30, activation='sigmoid')
    layer2f2 = layer2f1(layer1) # f for fixed
    layer2f1.trainable=False
    layer2t1 = tf.keras.layers.Dense(48, activation='sigmoid')
    layer2t2 = layer2t1(layer1) # t for trainable
    layer2 = tf.keras.layers.Concatenate()([layer2f2, layer2t2])
    layer3 = tf.keras.layers.Dense(1, activation='sigmoid')(layer2)
    _model = tf.keras.Model( inputs, layer3 )
    return _model, layer2f1.get_weights(), layer2t1.get_weights(), layer2t1, layer2f1
model1, init_f_weights_1,init_t_weights_1, layer2t1_1, layer2f1_1 = make_model1()
model1.summary()

def make_model2():
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    layer1 = tf.keras.layers.Dense( 64, activation='sigmoid')(inputs)
    layer2t1 = tf.keras.layers.Dense(48, activation='sigmoid')
    layer2 = layer2t1(layer1) # t for trainable
    layer3 = tf.keras.layers.Dense(1, activation='sigmoid')(layer2)
    _model = tf.keras.Model( inputs, layer3 )
    return _model, layer2t1.get_weights(), layer2t1
model2, init_t_weights_2, layer2t1_2 = make_model2()
model2.summary()


#------ run models ------
loss_fn = tf.keras.losses.MeanSquaredError()
loss_fn = tf.keras.losses.Huber()
model1.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0003) ,
        loss=loss_fn, metrics=['accuracy'])
history1=model1.fit(x_train, y_train, epochs=2000, verbose=1)

model2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003) ,
    loss=loss_fn, metrics=['accuracy'])
history2=model2.fit(x_train, y_train, epochs=2000, verbose=1)
plt.plot(history1.history['loss'])
plt.show()
plt.plot(history2.history['loss'])
plt.show()

if PRINT_WEIGHTS:
    print('*'*33)
    print ('model 1 initial trainable weights:')
    print(init_t_weights_1[0])
    print ('model 1 end trainable weights:')
    print(layer2t1_1.get_weights()[0])
    print ('difference between initial & end trainable weights:')
    print(init_t_weights_1[0] - layer2t1_1.get_weights()[0])
    print ('difference between initial & end untrainable weights:')
    print(init_f_weights_1[0] - layer2f1_1.get_weights()[0])
    print('*'*33)

    print('*'*33)
    print ('model 2 initial trainable weights:')
    print(init_t_weights_1[0])
    print ('model 2 end trainable weights:')
    print(layer2t1_1.get_weights()[0])
    



#------- plot histogram ------
if PLOT_WEIGHTS_HISTOGRAM :
    print('Plotting...')
    d_hist = {}
    model_diffs = np.ndarray.flatten(np.array(
            (layer2t1_1.get_weights()[0])))
    plt.hist(model_diffs)
    plt.show()
    model_diffs = np.ndarray.flatten(np.array(
            (init_t_weights_1[0] - layer2t1_1.get_weights()[0])))
    X = list(model_diffs )
    for xi in X:
        if xi in d_hist: d_hist[xi]+=1
        else: d_hist[xi]=1

    d_arr = [ d_hist[k] for k in sorted(d_hist.keys())]
    plt.scatter(d_hist.keys(), d_arr)
#    plt.rcParams["figure.figsize"] = (10, 20) # (w, h)
    plt.yscale('log')
    plt.show()
    print('Done plotting.')


if SANITY:
    print('>> y_true ', y_train[:1])
    print('>> y_pred ', model2(x_train[:1]))
    print('>> mse    ',loss_fn(y_train[:1], model2(x_train[:1]).numpy()))

# test:
if OLD_TEST:
#    test_input = tf.ones(shape=[BATCHSIZE]+list(INPUT_SHAPE))
    test_input = tf.random.truncated_normal(shape=[1]+list(INPUT_SHAPE))
#    print(test_input.numpy())
    test_output = model(test_input)
    print('input:                ', test_input)
    print('output:               ', test_output)
print('\t <><> Exiting Script <><>')
