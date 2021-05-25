#  See for original code (incl. code for saving):
#  /Users/morganbryant/Desktop/Files/21/01_January/passio/code-challenge/...
print('\t <><> Beginning Script <><>')
import tensorflow as tf
import os, sys

BATCHSIZE=8
#INPUT_SHAPE = (BATCHSIZE, 224,224,3)
INPUT_SHAPE = (13,)
TEST, SAVE = True, False    # What to do with the constructed model


(x_train, y_train), (x_test, y_test) =\
        tf.keras.datasets.boston_housing.load_data(test_split=0.2, seed=113)

print('*'*33)
#inputs = tf.keras.Input(shape=INPUT_SHAPE)
##inp_layer = tf.random.normal(INPUT_SHAPE)
#layer1 = tf.keras.layers.Conv2D( 16, 5, activation='relu', \
#        input_shape=INPUT_SHAPE)(inputs)
inputs = tf.keras.Input(shape=INPUT_SHAPE, batch_size=BATCHSIZE)

#layer1 = tf.keras.layers.Dense( 16, activation='relu')
#layer2a = tf.keras.layers.Dense(32, activation='relu')
#layer2b = tf.keras.layers.Dense(24, activation='relu')
#layer2b.trainable=False
#layer3 = tf.keras.layers.Dense(64, activation='relu')([layer2a, layer2b])
#layer4 = tf.keras.layers.Dense(2, activation='tanh')(layer3)
#model = tf.keras.Model( inputs, outputs=layer4(layer3(layer2(layer1(inputs)))) )
#sys.exit()

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

sys.exit()


#model = tf.keras.models.Sequential([
#    tf.keras.layers.Flatten(input_shape=INPUT_SHAPE),
#    tf.keras.layers.concatenate([\
#        tf.keras.layers.Dense(32, trainable=True, activation='relu'),
#        tf.keras.layers.Dense(24, trainable=False, activation='relu')
#    ]),
#    tf.keras.layers.Dense(64),
#    tf.keras.layers.Dense(2),
#])

predictions = model(x_train[:1]).numpy()
print(predictions)
from time import sleep
sleep(4)

#intermediate_model = base_model(inputs, training=False) 
#intermediate_model.trainable = False
#intermediate_model = tf.keras.layers.GlobalAveragePooling2D()(intermediate_model)
#intermediate_model = intermediate_model/tf.norm(intermediate_model, ord='euclidean')
#
#model = tf.keras.Model( inputs, intermediate_model )



# test:
if TEST:
    test_input = tf.ones((1,224,224,3))
    test_input = tf.ones(INPUT_SHAPE)
    test_output = model(test_input)
    print('input:                ', '<<tf.ones('+str(INPUT_SHAPE)+')>>')
    print('output:               ', test_output)
    print('sum of output:        ', tf.math.reduce_sum(test_output))
    print('l2-normalized output: ', tf.math.l2_normalize(test_output ))
    print('l2-length of output:  ', tf.norm(test_output))
print('\t <><> Exiting Script <><>')
