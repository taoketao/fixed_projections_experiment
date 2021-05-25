#  See for original code (incl. code for saving):
#  /Users/morganbryant/Desktop/Files/21/01_January/passio/code-challenge/...
print('\t <><> Beginning Script <><>')
import tensorflow as tf
import os, sys

BATCHSIZE=12
INPUT_SHAPE = (13,)
OLD_TEST, DEV = False, False    # What to do with the constructed model


data = tf.keras.datasets.boston_housing.load_data(test_split=0.2, seed=113)
x_train, y_train, x_test, y_test = \
        [tf.convert_to_tensor(d) for d in \
        (data[0][0], data[0][1], data[1][0], data[1][1])]
print (data[0][0].shape, data[0][1].shape, data[1][0].shape, data[1][1].shape, )

print('*'*33)
#------ build model ------
#inputs = tf.keras.Input(shape=INPUT_SHAPE, batch_size=BATCHSIZE)
def model1():
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    layer1 = tf.keras.layers.Dense( 10, activation='sigmoid')(inputs)
    layer2f1 = tf.keras.layers.Dense(3, activation='sigmoid')
    layer2f2 = layer2f1(layer1) # f for fixed
#layer2f2.trainable=False
    layer2f1.trainable=False
    layer2t1 = tf.keras.layers.Dense(6, activation='sigmoid')
    layer2t2 = layer2t1(layer1) # t for trainable
    layer2 = tf.keras.layers.Concatenate()([layer2f2, layer2t2])
    layer3 = tf.keras.layers.Dense(1, activation='sigmoid')(layer2)
    _model = tf.keras.Model( inputs, layer3 )
#    initial_layer2f_weights_values = layer2f1.get_weights()
#    initial_layer2t_weights_values = layer2t1.get_weights()
#    ___trainable, ___untrainable = initial_layer2f_weights_values
    return _model, layer2f1.get_weights(), layer2t1.get_weights(), layer2t1, layer2f1
model, init_f_weights,init_t_weights, layer2t1, layer2f1 = model1()

def model2():
    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    layer1 = tf.keras.layers.Dense( 10, activation='sigmoid')(inputs)
    layer2t1 = tf.keras.layers.Dense(6, activation='sigmoid')
    layer2t2 = layer2t1(layer1) # t for trainable
    layer2 = layer2t2
    layer3 = tf.keras.layers.Dense(1, activation='sigmoid')(layer2)
    _model = tf.keras.Model( inputs, layer3 )
#    initial_layer2f_weights_values = layer2f1.get_weights()
#    initial_layer2t_weights_values = layer2t1.get_weights()
#    ___trainable, ___untrainable = initial_layer2f_weights_values
    return _model, layer2t1.get_weights(), layer2t1
model, init_t_weights, layer2t1 = model2()




model.summary()

#------ build model ------
loss_fn = tf.keras.losses.MeanSquaredError()
#loss_fn = tf.keras.losses.SparseCategoricalCrossEntropy()
print('>> y_true ', y_train[:1])
print('>> y_pred ', model(x_train[:1]))
print('>> mse    ',loss_fn(y_train[:1], model(x_train[:1]).numpy()))
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1) ,
        loss=loss_fn,
        metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)
print('*'*33)
print ('initial trainable weights:')
print(init_t_weights[0])
print ('end trainable weights:')
print(layer2t1.get_weights()[0])
print ('difference between initial & end trainable weights:')
print(init_t_weights[0] - layer2t1.get_weights()[0])
print ('difference between initial & end untrainable weights:')
print(init_f_weights[0] - layer2f1.get_weights()[0])
print('*'*33)





# test:
if OLD_TEST:
#    test_input = tf.ones(shape=[BATCHSIZE]+list(INPUT_SHAPE))
    test_input = tf.random.truncated_normal(shape=[1]+list(INPUT_SHAPE))
#    print(test_input.numpy())
    test_output = model(test_input)
    print('input:                ', test_input)
    print('output:               ', test_output)
print('\t <><> Exiting Script <><>')
