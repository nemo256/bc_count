##############################################
#                                            #
#                 DO-U-Net                   #
#                   and                      #
#                 DO-SegNet                  #
#                                            #
# Author: Amine Neggazi                      #
# Email: neggazimedlamine@gmail/com          #
# Nick: nemo256                              #
#                                            #
# Please read bc-count/LICENSE               #
#                                            #
##############################################

import tensorflow as tf
import tensorflow_addons as tfa

# custom imports
from config import *


def conv_bn(filters,
            model,
            model_type,
            kernel=(3, 3),
            activation='relu', 
            strides=(1, 1),
            padding='valid',
            type='normal'):
    '''
    This is a custom convolution function:
    :param filters --> number of filters for each convolution
    :param kernel --> the kernel size
    :param activation --> the general activation function (relu)
    :param strides --> number of strides
    :param padding --> model padding (can be valid or same)
    :param type --> to indicate if it is a transpose or normal convolution

    :return --> returns the output after the convolution and batch normalization and activation.
    '''
    if model_type == 'segnet':
        kernel=3
        activation='relu'
        strides=(1, 1)
        padding='same'
        type='normal'

    if type == 'transpose':
        kernel = (2, 2)
        strides = 2
        conv = tf.keras.layers.Conv2DTranspose(filters, kernel, strides, padding)(model)
    else:
        conv = tf.keras.layers.Conv2D(filters, kernel, strides, padding)(model)

    conv = tf.keras.layers.BatchNormalization()(conv)
    conv = tf.keras.layers.Activation(activation)(conv)

    return conv


def max_pool(input):
    '''
    This is a general max pool function with custom parameters.
    '''
    return tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)(input)


def concatenate(input1, input2, crop):
    '''
    This is a general concatenation function with custom parameters.
    '''
    return tf.keras.layers.concatenate([tf.keras.layers.Cropping2D(crop)(input1), input2])


def get_callbacks(name):
    '''
    This is a custom function to save only the best checkpoint.
    :param name --> the input model name
    '''
    return [
        tf.keras.callbacks.ModelCheckpoint(f'models/{name}.h5',
                                           save_best_only=True,
                                           save_weights_only=True,
                                           verbose=1)
    ]


def do_unet():
    '''
    This is the dual output U-Net model.
    It is a custom U-Net with optimized number of layers.
    Please read model.summary()
    '''
    inputs = tf.keras.layers.Input((188, 188, 3))

    # encoder
    filters = 32
    encoder1 = conv_bn(3*filters, inputs, model_type)
    encoder1 = conv_bn(filters, encoder1, model_type, kernel=(1, 1))
    encoder1 = conv_bn(filters, encoder1, model_type)
    pool1 = max_pool(encoder1)

    filters *= 2
    encoder2 = conv_bn(filters, pool1, model_type)
    encoder2 = conv_bn(filters, encoder2, model_type)
    pool2 = max_pool(encoder2)

    filters *= 2
    encoder3 = conv_bn(filters, pool2, model_type)
    encoder3 = conv_bn(filters, encoder3, model_type)
    pool3 = max_pool(encoder3)

    filters *= 2
    encoder4 = conv_bn(filters, pool3, model_type)
    encoder4 = conv_bn(filters, encoder4, model_type)

    # decoder
    filters /= 2
    decoder1 = conv_bn(filters, encoder4, model_type, type='transpose')
    decoder1 = concatenate(encoder3, decoder1, 4)
    decoder1 = conv_bn(filters, decoder1, model_type)
    decoder1 = conv_bn(filters, decoder1, model_type)

    filters /= 2
    decoder2 = conv_bn(filters, decoder1, model_type, type='transpose')
    decoder2 = concatenate(encoder2, decoder2, 16)
    decoder2 = conv_bn(filters, decoder2, model_type)
    decoder2 = conv_bn(filters, decoder2, model_type)

    filters /= 2
    decoder3 = conv_bn(filters, decoder2, model_type, type='transpose')
    decoder3 = concatenate(encoder1, decoder3, 40)
    decoder3 = conv_bn(filters, decoder3, model_type)
    decoder3 = conv_bn(filters, decoder3, model_type)

    out_mask = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='mask')(decoder3)

    if cell_type == 'rbc':
        out_edge = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='edge')(decoder3)
        model = tf.keras.models.Model(inputs=inputs, outputs=(out_mask, out_edge))
    elif cell_type == 'wbc' or cell_type == 'plt':
        model = tf.keras.models.Model(inputs=inputs, outputs=(out_mask))

    opt = tf.optimizers.Adam(learning_rate=0.0001)

    if cell_type == 'rbc':
        model.compile(loss='mse',
                      loss_weights=[0.1, 0.9],
                      optimizer=opt,
                      metrics=['accuracy'])
    elif cell_type == 'wbc' or cell_type == 'plt':
        model.compile(loss='mse',
                      optimizer=opt,
                      metrics='accuracy')
    return model

def segnet():
    inputs = tf.keras.layers.Input((128, 128, 3))

    # encoder
    filters = 64
    encoder1 = conv_bn(filters, inputs, model_type)
    encoder1 = conv_bn(filters, encoder1, model_type)
    pool1, mask1 = tf.nn.max_pool_with_argmax(encoder1, 3, 2, padding="SAME")

    filters *= 2
    encoder2 = conv_bn(filters, pool1, model_type)
    encoder2 = conv_bn(filters, encoder2, model_type)
    pool2, mask2 = tf.nn.max_pool_with_argmax(encoder2, 3, 2, padding="SAME")

    filters *= 2
    encoder3 = conv_bn(filters, pool2, model_type)
    encoder3 = conv_bn(filters, encoder3, model_type)
    encoder3 = conv_bn(filters, encoder3, model_type)
    pool3, mask3 = tf.nn.max_pool_with_argmax(encoder3, 3, 2, padding="SAME")

    filters *= 2
    encoder4 = conv_bn(filters, pool3, model_type)
    encoder4 = conv_bn(filters, encoder4, model_type)
    encoder4 = conv_bn(filters, encoder4, model_type)
    pool4, mask4 = tf.nn.max_pool_with_argmax(encoder4, 3, 2, padding="SAME")

    encoder5 = conv_bn(filters, pool4, model_type)
    encoder5 = conv_bn(filters, encoder5, model_type)
    encoder5 = conv_bn(filters, encoder5, model_type)
    pool5, mask5 = tf.nn.max_pool_with_argmax(encoder5, 3, 2, padding="SAME")

    # decoder
    unpool1 = tfa.layers.MaxUnpooling2D()(pool5, mask5)
    decoder1 = conv_bn(filters, unpool1, model_type)
    decoder1 = conv_bn(filters, decoder1, model_type)
    decoder1 = conv_bn(filters, decoder1, model_type)

    unpool2 = tfa.layers.MaxUnpooling2D()(decoder1, mask4)
    decoder2 = conv_bn(filters, unpool2, model_type)
    decoder2 = conv_bn(filters, decoder2, model_type)
    decoder2 = conv_bn(filters/2, decoder2, model_type)

    filters /= 2
    unpool3 = tfa.layers.MaxUnpooling2D()(decoder2, mask3)
    decoder3 = conv_bn(filters, unpool3, model_type)
    decoder3 = conv_bn(filters, decoder3, model_type)
    decoder3 = conv_bn(filters/2, decoder3, model_type)

    filters /= 2
    unpool4 = tfa.layers.MaxUnpooling2D()(decoder3, mask2)
    decoder4 = conv_bn(filters, unpool4, model_type)
    decoder4 = conv_bn(filters/2, decoder4, model_type)

    filters /= 2
    unpool5 = tfa.layers.MaxUnpooling2D()(decoder4, mask1)
    decoder5 = conv_bn(filters, unpool5, model_type)

    out_mask = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='mask')(decoder5)

    if cell_type == 'rbc':
        out_edge = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='edge')(decoder5)
        model = tf.keras.models.Model(inputs=inputs, outputs=(out_mask, out_edge))
    elif cell_type == 'wbc' or cell_type == 'plt':
        model = tf.keras.models.Model(inputs=inputs, outputs=(out_mask))

    opt = tf.optimizers.Adam(learning_rate=0.0001)

    if cell_type == 'rbc':
        model.compile(loss='mse',
                      loss_weights=[0.1, 0.9],
                      optimizer=opt,
                      metrics='accuracy')
    elif cell_type == 'wbc' or cell_type == 'plt':
        model.compile(loss='mse',
                      optimizer=opt,
                      metrics='accuracy')
    return model
