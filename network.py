#  See for original code (incl. code for saving):
#  /Users/morganbryant/Desktop/Files/21/01_January/passio/code-challenge/...
print('\t <><> Beginning Script <><>')
import tensorflow as tf
import os, sys

BATCHSIZE=8
INPUT_SHAPE = (13,)
TEST, SAVE = True, False    # What to do with the constructed model


(x_train, y_train), (x_test, y_test) = data = \
        tf.keras.datasets.boston_housing.load_data(test_split=0.2, seed=113)
print (data[0][0].shape, data[0][1].shape, data[1][0].shape, data[1][1].shape, )

print('*'*33)
inputs = tf.keras.Input(shape=INPUT_SHAPE, batch_size=BATCHSIZE)
layer1 = tf.keras.layers.Dense( 16, activation='relu')(inputs)
layer2f1 = tf.keras.layers.Dense(8, activation='relu')
layer2f2 = layer2f1(layer1) # f for fixed
layer2f2.trainable=False
layer2t1 = tf.keras.layers.Dense(32, activation='relu')
layer2t2 = layer2t1(layer1) # t for trainable
layer2 = tf.keras.layers.Concatenate()([layer2f2, layer2t2])
layer3 = tf.keras.layers.Dense(64, activation='relu')(layer2)
layer4 = tf.keras.layers.Dense(2, activation='tanh')(layer3)
model = tf.keras.Model( inputs, layer4 )
initial_layer2f_weights_values = layer2f1.get_weights()
___trainable, ___untrainable = initial_layer2f_weights_values
print (initial_layer2f_weights_values[0].shape,initial_layer2f_weights_values[1].shape )
model.summary()


# test:
if TEST:
#    test_input = tf.ones(shape=[BATCHSIZE]+list(INPUT_SHAPE))
    test_input = tf.random.truncated_normal(shape=[BATCHSIZE]+list(INPUT_SHAPE))
#    print(test_input.numpy())
    test_output = model(test_input)
    print('input:                ', test_input)
    print('output:               ', test_output)
print('\t <><> Exiting Script <><>')
