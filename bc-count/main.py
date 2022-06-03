import glob
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import ndimage

# custom imports
import data
from model import do_unet, get_callbacks


# global variables
cell_type       = 'red'              # red, white or platelets
input_shape     = (188, 188, 3)
output_shape    = (100, 100, 1)
padding         = [200, 100]


def generate_train_dataset(img_list, mask_list, edge_list):
    img, mask, edge = data.load_data(img_list, mask_list, edge_list)

    def train_gen():
        return data.train_generator(img, mask, edge,
                                    padding=padding[0],
                                    input_size=input_shape[0],
                                    output_size=output_shape[0])

    # load train dataset to tensorflow for training
    return tf.data.Dataset.from_generator(
        train_gen,
        (tf.float64, ((tf.float64), (tf.float64))),
        (input_shape, (output_shape, output_shape))
    )


def generate_test_dataset(img_list, mask_list, edge_list):
    img, mask, edge = data.load_data(img_list, mask_list, edge_list)

    img_chips, mask_chips, edge_chips = data.test_chips(
        img,
        mask,
        edge,
        padding=padding[1],
        input_size=input_shape[0],
        output_size=output_shape[0]
    )

    # load test dataset to tensorflow for training
    return tf.data.Dataset.from_tensor_slices(
        (img_chips, (mask_chips, edge_chips))
    )


def train(model_name='mse', epochs=100):
    # globing appropriate images, their masks and their edges
    if cell_type == 'red':
        train_img_list = glob.glob('data/rbc/train/image/*.jpg')
        test_img_list = glob.glob('data/rbc/test/image/*.jpg')
        train_mask_list = glob.glob('data/rbc/train/mask/*.jpg')
        train_edge_list = glob.glob('data/rbc/train/edge/*.jpg')
        test_mask_list = glob.glob('data/rbc/test/mask/*.jpg')
        test_edge_list = glob.glob('data/rbc/test/edge/*.jpg')
    elif cell_type == 'white':
        train_img_list = glob.glob('data/wbc/train/image/*.jpg')
        test_img_list = glob.glob('data/wbc/test/image/*.jpg')
        train_mask_list = glob.glob('data/wbc/train/mask/*.jpg')
        train_edge_list = glob.glob('data/wbc/train/edge/*.jpg')
        test_mask_list = glob.glob('data/wbc/test/mask/*.jpg')
        test_edge_list = glob.glob('data/wbc/test/edge/*.jpg')
    else:
        train_mask_list = None
        test_mask_list = None

    # loading train dataset and test datasets
    train_dataset = generate_train_dataset(
        train_img_list,
        train_mask_list,
        train_edge_list
    )
    test_dataset = generate_test_dataset(
        test_img_list,
        test_mask_list,
        test_edge_list
    )

    # initializing the segnet model
    model = do_unet()

    # create models directory if it does not exist
    if not os.path.exists('models/'):
        os.makedirs('models/')

    # Check for existing weights
    if os.path.exists(f'models/{model_name}.h5'):
        model.load_weights(f'models/{model_name}.h5')

    # fitting the model
    history = model.fit(
        train_dataset.batch(8),
        validation_data=test_dataset.batch(8),
        epochs=epochs,
        steps_per_epoch=125,
        max_queue_size=16,
        use_multiprocessing=False,
        workers=8,
        verbose=1,
        callbacks=get_callbacks(model_name)
    )

    # save the history
    np.save(f'models/{model_name}_history.npy', history.history)


if __name__ == '__main__':
    train()
