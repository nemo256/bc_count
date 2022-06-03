import matplotlib.pyplot as plt
from scipy import ndimage

# custom imports
import data
from model import segnet, get_callbacks


# global variables
cell_type       = 'red'              # red, white or platelets
input_shape     = (128, 128, 3)
output_shape    = (128, 128, 1)
padding         = [200, 100]


def generate_train_dataset(img_list, mask_list):
    img, mask = data.load_data(img_list, mask_list)

    def train_gen():
        return data.train_generator(img, mask,
                                    padding=padding[0],
                                    input_size=input_shape[0],
                                    output_size=output_shape[0])

    # load test dataset to tensorflow for training
    return tf.data.Dataset.from_generator(
        train_gen,
        (tf.float64, ((tf.float64), (tf.float64))),
        (input_shape, (output_shape, output_shape))
    )


def generate_test_dataset(img_list, mask_list):
    img, mask = data.load_data(img_list, mask_list)

    img_chips, mask_chips = data.test_chips(
        img,
        mask,
        padding=padding[1],
        input_size=input_shape[0],
        output_size=output_shape[0]
    )

    # load test dataset to tensorflow for training
    return tf.data.Dataset.from_tensor_slices(
        (img_chips, (mask_chips, None))
    )
