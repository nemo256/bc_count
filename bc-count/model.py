import tensorflow as tf
import tensorflow_addons as tfa


def conv_bn(filters,
            model,
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

    returns the output after the convolutions.
    '''
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
    encoder1 = conv_bn(3*filters, inputs)
    encoder1 = conv_bn(filters, encoder1, kernel=(1, 1))
    encoder1 = conv_bn(filters, encoder1)
    pool1 = max_pool(encoder1)

    filters *= 2
    encoder2 = conv_bn(filters, pool1)
    encoder2 = conv_bn(filters, encoder2)
    pool2 = max_pool(encoder2)

    filters *= 2
    encoder3 = conv_bn(filters, pool2)
    encoder3 = conv_bn(filters, encoder3)
    pool3 = max_pool(encoder3)

    filters *= 2
    encoder4 = conv_bn(filters, pool3)
    encoder4 = conv_bn(filters, encoder4)

    # decoder
    filters /= 2
    decoder1 = conv_bn(filters, encoder4, type='transpose')
    decoder1 = concatenate(encoder3, decoder1, 4)
    decoder1 = conv_bn(filters, decoder1)
    decoder1 = conv_bn(filters, decoder1)

    filters /= 2
    decoder2 = conv_bn(filters, decoder1, type='transpose')
    decoder2 = concatenate(encoder2, decoder2, 16)
    decoder2 = conv_bn(filters, decoder2)
    decoder2 = conv_bn(filters, decoder2)

    filters /= 2
    decoder3 = conv_bn(filters, decoder2, type='transpose')
    decoder3 = concatenate(encoder1, decoder3, 40)
    decoder3 = conv_bn(filters, decoder3)
    decoder3 = conv_bn(filters, decoder3)

    out_mask = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='mask')(decoder3)
    out_edge = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid', name='edge')(decoder3)

    model = tf.keras.models.Model(inputs=inputs, outputs=(out_mask, out_edge))

    opt = tf.optimizers.Adam(learning_rate=0.0001)

    model.compile(loss='mse',
                  loss_weights=[0.1, 0.9],
                  optimizer=opt,
                  metrics='accuracy')

    return model
