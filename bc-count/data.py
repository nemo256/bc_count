##############################################
#                                            #
#           Custom data generator            #
#                                            #
# Author: Amine Neggazi                      #
# Email: neggazimedlamine@gmail/com          #
# Nick: nemo256                              #
#                                            #
# Please read bc-count/LICENSE               #
#                                            #
##############################################

import os
import json

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

# custom imports
from config import *


def load_image_list(img_files, gray=False):
    '''
    This is the load image list function, which loads an enumerate
    of images (param: img_files)
    :param img_files --> the input image files which we want to read

    :return imgs --> the images that we read
    '''
    imgs = []
    if gray:
        for image_file in img_files:
            img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
            imgs += [img]

    else:
        for image_file in img_files:
            imgs += [cv2.imread(image_file)]
    return imgs


def clahe_images(img_list):
    '''
    This is the clahe images function, which applies a clahe threshold
    the input image list.
    :param img_files --> the input image files which we want to read

    :return img_list --> the output images
    '''
    for i, img in enumerate(img_list):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[..., 0] = clahe.apply(lab[..., 0])
        img_list[i] = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return img_list


def preprocess_image(imgs, padding=padding[1]):
    '''
    This is the preprocess data function, which adds a padding to 
    the input images, masks and edges if there are any.
    :param imgs --> the input list of images.
    :param padding --> the input padding which is going to be applied.

    :return imgs --> output images with added padding.
    '''
    imgs = [np.pad(img, ((padding, padding),
                         (padding, padding), (0, 0)), mode='constant') for img in imgs]
    return imgs


def preprocess_data(imgs, mask, edge=None, padding=padding[1]):
    '''
    This is the preprocess data function, which adds a padding to 
    the input images, masks and edges if there are any.
    :param imgs --> the input list of images.
    :param mask --> the input list of masks.
    :param edge --> the input list of edges.
    :param padding --> the input padding which is going to be applied.

    :return tuple(imgs, mask, edge if exists) --> output images, masks and edges with padding added.
    '''
    imgs = [np.pad(img, ((padding, padding),
                         (padding, padding), (0, 0)), mode='constant') for img in imgs]
    mask = [np.pad(mask, ((padding, padding),
                          (padding, padding)), mode='constant') for mask in mask]
    if edge is not None:
        edge = [np.pad(edge, ((padding, padding),
                              (padding, padding)), mode='constant') for edge in edge]

    if edge is not None:
        return imgs, mask, edge

    return imgs, mask


def load_data(img_list, mask_list, edge_list=None, padding=padding[1]):
    '''
    This is the load data function, which will handle image loading and preprocessing.
    :param img_list --> list of input images
    :param mask_list --> list of input masks
    :param edge_list --> list of input edges
    :param padding --> padding to be applied on preprocessing

    :return tuple(imgs, masks and edges if exists) --> the output preprocessed imgs, masks and edges.
    '''
    imgs = load_image_list(img_list)
    imgs = clahe_images(imgs)

    mask = load_image_list(mask_list, gray=True)
    if edge_list:
        edge = load_image_list(edge_list, gray=True)
    else:
        edge = None

    return preprocess_data(imgs, mask, edge, padding=padding)


def load_image(img_list, padding=padding[1]):
    '''
    This is the load data function, which will handle image loading and preprocessing.
    :param img_list --> list of input images
    :param padding --> padding to be applied on preprocessing

    :return imgs --> the output preprocessed imgs.
    '''
    imgs = load_image_list(img_list)
    imgs = clahe_images(imgs)
    return preprocess_image(imgs, padding=padding)


def aug_lum(image, factor=None):
    '''
    This is the augment luminosity function, which we apply to
    augment the luminosity of an input image.
    :param image --> the input image we want to augment
    :param factor --> the factor of luminosity augment (default is 0.5 * random number)

    :return image --> the output luminosity augmented image
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float64)

    if factor is None:
        lum_offset = 0.5 + np.random.uniform()
    else:
        lum_offset = factor

    hsv[..., 2] = hsv[..., 2] * lum_offset
    hsv[..., 2][hsv[..., 2] > 255] = 255
    hsv = hsv.astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def aug_img(image):
    '''
    This is the augment colors function, which we apply to
    augment the colors of an given image.
    :param image --> the input image we want to augment

    :return image --> the output colors augmented image
    '''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype(np.float64)

    hue_offset = 0.8 + 0.4*np.random.uniform()
    sat_offset = 0.5 + np.random.uniform()
    lum_offset = 0.5 + np.random.uniform()

    hsv[..., 0] = hsv[..., 0] * hue_offset
    hsv[..., 1] = hsv[..., 1] * sat_offset
    hsv[..., 2] = hsv[..., 2] * lum_offset

    hsv[..., 0][hsv[..., 0] > 255] = 255
    hsv[..., 1][hsv[..., 1] > 255] = 255
    hsv[..., 2][hsv[..., 2] > 255] = 255

    hsv = hsv.astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def train_generator(imgs, mask, edge=None,
                    scale_range=None,
                    padding=padding[1],
                    input_size=input_shape[0],
                    output_size=output_shape[0],
                    skip_empty=False):
    '''
    This is the train generator function, which generates the train dataset.
    :param imgs --> the input images
    :param mask --> the input masks
    :param edge --> the input edges if there are any (red blood cells only)
    :param scale_range --> the factor (i, j) of rescaling.
    :param padding --> the padding which will be applied to each image
    :param input_size --> the input shape
    :param output_size --> the output shape
    :param skip_empty --> skip empty chips (random if not set)

    :return chips --> yields an image, mask and edge chip each time it gets executed (called)
    '''
    if scale_range is not None:
        scale_range = [1 - scale_range, 1 + scale_range]
    while True:
        # select which type of cell to return
        chip_type = np.random.choice([True, False])

        while True:
            # pick random image
            i = np.random.randint(len(imgs))

            # pick random central location in the image (200 + 196/2)
            center_offset = padding + (output_size / 2)
            x = np.random.randint(center_offset, imgs[i].shape[0] - center_offset)
            y = np.random.randint(center_offset, imgs[i].shape[1] - center_offset)

            # scale the box randomly from x0.8 - 1.2x original size
            scale = 1
            if scale_range is not None:
                scale = scale_range[0] + ((scale_range[0] - scale_range[0]) * np.random.random())

            # find the edges of a box around the image chip and the mask chip
            chip_x_l = int(x - ((input_size / 2) * scale))
            chip_x_r = int(x + ((input_size / 2) * scale))
            chip_y_l = int(y - ((input_size / 2) * scale))
            chip_y_r = int(y + ((input_size / 2) * scale))

            mask_x_l = int(x - ((output_size / 2) * scale))
            mask_x_r = int(x + ((output_size / 2) * scale))
            mask_y_l = int(y - ((output_size / 2) * scale))
            mask_y_r = int(y + ((output_size / 2) * scale))

            # take a slice of the image and mask accordingly
            temp_chip = imgs[i][chip_x_l:chip_x_r, chip_y_l:chip_y_r]
            temp_mask = mask[i][mask_x_l:mask_x_r, mask_y_l:mask_y_r]
            if edge is not None:
                temp_edge = edge[i][mask_x_l:mask_x_r, mask_y_l:mask_y_r]

            if skip_empty:
                if ((temp_mask > 0).sum() > 5) is chip_type:
                    continue

            # resize the image chip back to 380 and the mask chip to 196
            temp_chip = cv2.resize(temp_chip,
                                   (input_size, input_size),
                                   interpolation=cv2.INTER_CUBIC)
            temp_mask = cv2.resize(temp_mask,
                                   (output_size, output_size),
                                   interpolation=cv2.INTER_NEAREST)
            if edge is not None:
                temp_edge = cv2.resize(temp_edge,
                                       (output_size, output_size),
                                       interpolation=cv2.INTER_NEAREST)

            # randomly rotate (like below)
            rot = np.random.randint(4)
            temp_chip = np.rot90(temp_chip, k=rot, axes=(0, 1))
            temp_mask = np.rot90(temp_mask, k=rot, axes=(0, 1))
            if edge is not None:
                temp_edge = np.rot90(temp_edge, k=rot, axes=(0, 1))

            # randomly flip
            if np.random.random() > 0.5:
                temp_chip = np.flip(temp_chip, axis=1)
                temp_mask = np.flip(temp_mask, axis=1)
                if edge is not None:
                    temp_edge = np.flip(temp_edge, axis=1)

            # randomly luminosity augment
            temp_chip = aug_lum(temp_chip)

            # randomly augment chip
            temp_chip = aug_img(temp_chip)

            # rescale the image
            temp_chip = temp_chip.astype(np.float32) * 2
            temp_chip /= 255
            temp_chip -= 1

            # later on ... randomly adjust colours
            if edge is not None:
                yield temp_chip, ((temp_mask > 0).astype(float)[..., np.newaxis], 
                                  (temp_edge > 0).astype(float)[..., np.newaxis])
            else:
                yield temp_chip, ((temp_mask > 0).astype(float)[..., np.newaxis])
            break


def test_chips(imgs, mask,
               edge=None,
               padding=padding[1],
               input_size=input_shape[0],
               output_size=output_shape[0]):
    '''
    This is the test chips function, which generates the test dataset.
    :param imgs --> the input images
    :param mask --> the input masks
    :param edge --> the input edges if there are any (red blood cells only)
    :param padding --> the padding which will be applied to each image
    :param input_size --> the input shape
    :param output_size --> the output shape

    :return chips --> yields an image, mask and edge chip each time it gets executed (called)
    '''
    center_offset = padding + (output_size / 2)
    for i, _ in enumerate(imgs):
        for x in np.arange(center_offset, imgs[i].shape[0] - input_size / 2, output_size):
            for y in np.arange(center_offset, imgs[i].shape[1] - input_size / 2, output_size):
                chip_x_l = int(x - (input_size / 2))
                chip_x_r = int(x + (input_size / 2))
                chip_y_l = int(y - (input_size / 2))
                chip_y_r = int(y + (input_size / 2))

                mask_x_l = int(x - (output_size / 2))
                mask_x_r = int(x + (output_size / 2))
                mask_y_l = int(y - (output_size / 2))
                mask_y_r = int(y + (output_size / 2))

                temp_chip = imgs[i][chip_x_l:chip_x_r, chip_y_l:chip_y_r]
                temp_mask = mask[i][mask_x_l:mask_x_r, mask_y_l:mask_y_r]
                if edge is not None:
                    temp_edge = edge[i][mask_x_l:mask_x_r, mask_y_l:mask_y_r]

                temp_chip = temp_chip.astype(np.float32) * 2
                temp_chip /= 255
                temp_chip -= 1

                if edge is not None:
                    yield temp_chip, ((temp_mask > 0).astype(float)[..., np.newaxis], 
                                      (temp_edge > 0).astype(float)[..., np.newaxis])
                else:
                    yield temp_chip, ((temp_mask > 0).astype(float)[..., np.newaxis])
                break


def slice_image(imgs,
                padding=padding[1],
                input_size=input_shape[0],
                output_size=output_shape[0]):
    '''
    This is the slice function, which slices each image into image chips.
    :param imgs --> the input images
    :param padding --> the padding which will be applied to each image
    :param input_size --> the input shape
    :param output_size --> the output shape

    :return list tuple (list, list, list) --> the tuple list of output (image, mask and edge chips)
    '''
    img_chips = []

    center_offset = padding + (output_size / 2)
    for i, _ in enumerate(imgs):
        for x in np.arange(center_offset, imgs[i].shape[0] - input_size / 2, output_size):
            for y in np.arange(center_offset, imgs[i].shape[1] - input_size / 2, output_size):
                chip_x_l = int(x - (input_size / 2))
                chip_x_r = int(x + (input_size / 2))
                chip_y_l = int(y - (input_size / 2))
                chip_y_r = int(y + (input_size / 2))

                temp_chip = imgs[i][chip_x_l:chip_x_r, chip_y_l:chip_y_r]

                temp_chip = temp_chip.astype(np.float32) * 2
                temp_chip /= 255
                temp_chip -= 1

                img_chips += [temp_chip]
    return np.array(img_chips)


def slice(imgs, mask,
          edge=None,
          padding=padding[1],
          input_size=input_shape[0],
          output_size=output_shape[0]):
    '''
    This is the slice function, which slices each image into image chips.
    :param imgs --> the input images
    :param mask --> the input masks
    :param edge --> the input edges if there are any (red blood cells only)
    :param padding --> the padding which will be applied to each image
    :param input_size --> the input shape
    :param output_size --> the output shape

    :return list tuple (list, list, list) --> the tuple list of output (image, mask and edge chips)
    '''
    img_chips = []
    mask_chips = []
    if edge is not None:
        edge_chips = []

    center_offset = padding + (output_size / 2)
    for i, _ in enumerate(imgs):
        for x in np.arange(center_offset, imgs[i].shape[0] - input_size / 2, output_size):
            for y in np.arange(center_offset, imgs[i].shape[1] - input_size / 2, output_size):
                chip_x_l = int(x - (input_size / 2))
                chip_x_r = int(x + (input_size / 2))
                chip_y_l = int(y - (input_size / 2))
                chip_y_r = int(y + (input_size / 2))

                mask_x_l = int(x - (output_size / 2))
                mask_x_r = int(x + (output_size / 2))
                mask_y_l = int(y - (output_size / 2))
                mask_y_r = int(y + (output_size / 2))

                temp_chip = imgs[i][chip_x_l:chip_x_r, chip_y_l:chip_y_r]
                temp_mask = mask[i][mask_x_l:mask_x_r, mask_y_l:mask_y_r]
                if edge is not None:
                    temp_edge = edge[i][mask_x_l:mask_x_r, mask_y_l:mask_y_r]

                temp_chip = temp_chip.astype(np.float32) * 2
                temp_chip /= 255
                temp_chip -= 1

                img_chips += [temp_chip]
                mask_chips += [(temp_mask > 0).astype(float)[..., np.newaxis]]
                if edge is not None:
                    edge_chips += [(temp_edge > 0).astype(float)[..., np.newaxis]]

    img_chips = np.array(img_chips)
    mask_chips = np.array(mask_chips)
    if edge is not None:
        edge_chips = np.array(edge_chips)

    if edge is not None:
        return img_chips, mask_chips, edge_chips

    return img_chips, mask_chips


def generator(img_list, mask_list, edge_list=None, type='train'):
    '''
    This is the generator function, which provides the list of image, mask and edge lists to the train generator and test chips functions.
    :param img_list --> the input list of images
    :param mask_list --> the input list of masks
    :param edge_list --> the input list of edges if there are any
    :param type --> can be either train or test, used to determine which generator function is to be called

    :return tensorflow dataset --> the output generated functions fed to tensorflow
    '''
    if cell_type == 'rbc':
        img, mask, edge = load_data(img_list, mask_list, edge_list)
    elif cell_type == 'wbc' or cell_type == 'plt':
        img, mask = load_data(img_list, mask_list)
        edge = None

    def gen():
        if type == 'train':
            return train_generator(img, mask, edge,
                                   padding=padding[0],
                                   input_size=input_shape[0],
                                   output_size=output_shape[0])
        elif type == 'test':
            return test_chips(img, mask, edge,
                              padding=padding[0],
                              input_size=input_shape[0],
                              output_size=output_shape[0])

    # load train dataset to tensorflow for training
    if cell_type == 'rbc':
        return tf.data.Dataset.from_generator(
            gen,
            (tf.float64, ((tf.float64), (tf.float64))),
            (input_shape, (output_shape, output_shape))
        )
    elif cell_type == 'wbc' or cell_type == 'plt':
        return tf.data.Dataset.from_generator(
            gen,
            (tf.float64, (tf.float64)),
            (input_shape, (output_shape))
        )
