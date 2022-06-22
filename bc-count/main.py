##############################################
#                                            #
# Main program (later this will be changed)  #
# for simplicities sake this will only call  # 
# function from algorithms/<>.py files       # 
#                                            #
# Author: Amine Neggazi                      #
# Email: neggazimedlamine@gmail/com          #
# Nick: nemo256                              #
#                                            #
# Please read bc-count/LICENSE               #
#                                            #
##############################################

import glob
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

# custom imports
from config import *
import data
from model import do_unet, segnet, get_callbacks


def train(model_name='mse', epochs=50):
    '''
    This is the train function, so that we can train multiple models
    according to blood cell types and multiple input shapes aswell.
    :param model_name --> model weights that we want saved
    :param epochs --> how many epochs we want the model to be trained

    :return --> saves the model weights under <model_name>.h5 with
    its respective history file <model_name>_history.npy
    '''

    train_img_list = sorted(glob.glob(f'data/{cell_type}/train/image/*.jpg'))
    test_img_list = sorted(glob.glob(f'data/{cell_type}/test/image/*.jpg'))
    train_mask_list = sorted(glob.glob(f'data/{cell_type}/train/mask/*.jpg'))
    test_mask_list = sorted(glob.glob(f'data/{cell_type}/test/mask/*.jpg'))

    if cell_type == 'rbc':
        train_edge_list = sorted(glob.glob(f'data/{cell_type}/train/edge/*.jpg'))
        test_edge_list = sorted(glob.glob(f'data/{cell_type}/test/edge/*.jpg'))
    elif cell_type == 'wbc' or cell_type == 'plt':
        train_edge_list = None
        test_edge_list = None
    else:
        print('Invalid blood cell type!\n')
        return

    # loading train dataset and test datasets
    train_dataset = data.generator(
        train_img_list,
        train_mask_list,
        train_edge_list,
        type='train'
    )
    test_dataset = data.generator(
        test_img_list,
        test_mask_list,
        test_edge_list,
        type='test'
    )

    # initializing the do_unet model
    if model_type == 'do_unet':
        model = do_unet()
    else:
        model = segnet()

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
        use_multiprocessing=True,
        workers=8,
        verbose=1,
        callbacks=get_callbacks(model_name)
    )

    # save the history
    np.save(f'models/{model_name}_history.npy', history.history)


def normalize(img):
    '''
    Normalizes an image
    :param img --> an input image that we want normalized

    :return np.array --> an output image normalized (as a numpy array)
    '''
    return np.array((img - np.min(img)) / (np.max(img) - np.min(img)))


def get_sizes(img,
              padding=padding[1],
              input=input_shape[0],
              output=output_shape[0]):
    '''
    Get full image sizes (x, y) to rebuilt the full image output
    :param img --> an input image we want to get its dimensions
    :param padding --> the default padding used on the test dataset
    :param input --> the input shape of the image (param: img)
    :param output --> the output shape of the image (param: img)

    :return couple --> a couple which contains the image dimensions as in (x, y)
    '''
    offset = padding + (output / 2)
    return [(len(np.arange(offset, img[0].shape[0] - input / 2, output)), len(np.arange(offset, img[0].shape[1] - input / 2, output)))]


def reshape(img,
            size_x,
            size_y):
    '''
    Reshape the full image output using the original sizes (x, y)
    :param img --> an input image we want to reshape
    :param size_x --> the x axis (length) of the input image (param: img)
    :param size_y --> the y axis (length) of the input image (param: img)

    :return img (numpy array) --> the output image reshaped according to the provided dimensions (size_x, size_y)
    '''
    return img.reshape(size_x, size_y, output_shape[0], output_shape[0], 1)


def concat(imgs):
    '''
    Concatenate all the output image chips to rebuild the full image
    :param imgs --> the images that we want to concatenate

    :return full_image --> the concatenation of all the provided images (param: imgs)
    '''
    return cv2.vconcat([cv2.hconcat(im_list) for im_list in imgs[:,:,:,:]])


def denoise(img):
    '''
    Remove noise from an image
    :param img --> the input image that we want to denoise (remove the noise)

    :return image --> the denoised output image
    '''
    # read the image
    img = cv2.imread(img)
    # return the denoised image
    return cv2.fastNlMeansDenoising(img, 23, 23, 7, 21)


def predict(imgName='Im037_0'):
    '''
    Predict (segment) blood cell images using the trained model (do_unet)
    :param img --> the image we want to predict (from the test/ directory)

    :return --> saves the predicted (segmented blood cell image) under the folder output/
    '''
    # Check for existing predictions
    if not os.path.exists(f'{output_directory}/{imgName}'):
        os.makedirs(f'{output_directory}/{imgName}', exist_ok=True)
    else:
        print('Prediction already exists!')
        return

    test_img = sorted(glob.glob(f'data/ALL-IDB1-{cell_type}/{imgName}.jpg'))

    # initializing the do_unet model
    if model_type == 'do_unet':
        model = do_unet()
    else:
        model = segnet()

    # Check for existing weights
    if os.path.exists(f'models/{model_name}.h5'):
        model.load_weights(f'models/{model_name}.h5')

    # load test data
    img = data.load_image(test_img, padding=padding[0])

    img_chips = data.slice_image(
        img,
        padding=padding[1],
        input_size=input_shape[0],
        output_size=output_shape[0],
    )

    # segment all image chips
    output = model.predict(img_chips)

    if cell_type == 'rbc':
        new_mask_chips = np.array(output[0])
        new_edge_chips = np.array(output[1])
    elif cell_type == 'wbc' or cell_type == 'plt':
        new_mask_chips = np.array(output)

    # get image dimensions
    dimensions = [get_sizes(img)[0][0], get_sizes(img)[0][1]]

    # reshape chips arrays to be concatenated
    new_mask_chips = reshape(new_mask_chips, dimensions[0], dimensions[1])
    if cell_type == 'rbc':
        new_edge_chips = reshape(new_edge_chips, dimensions[0], dimensions[1])

    # get rid of none necessary dimension
    new_mask_chips = np.squeeze(new_mask_chips)
    if cell_type == 'rbc':
        new_edge_chips = np.squeeze(new_edge_chips)

    # concatenate chips into a single image (mask and edge)
    new_mask = concat(new_mask_chips)
    if cell_type == 'rbc':
        new_edge = concat(new_edge_chips)
    
    # save predicted mask and edge
    plt.imsave(f'{output_directory}/{imgName}/mask.png', new_mask)
    if cell_type == 'rbc':
        plt.imsave(f'{output_directory}/{imgName}/edge.png', new_edge)
        plt.imsave(f'{output_directory}/{imgName}/edge_mask.png', new_mask - new_edge)

    if model_type == 'segnet':
        # denoise all the output images
        new_mask  = denoise(f'{output_directory}/{imgName}/mask.png')
        if cell_type == 'rbc':
            new_edge  = denoise(f'{output_directory}/{imgName}/edge.png')
            edge_mask = denoise(f'{output_directory}/{imgName}/edge_mask.png')

        # save predicted mask and edge after denoising
        plt.imsave(f'{output_directory}/{imgName}/mask.png', new_mask)
        if cell_type == 'rbc':
            plt.imsave(f'{output_directory}/{imgName}/edge.png', new_edge)
            plt.imsave(f'{output_directory}/{imgName}/edge_mask.png', edge_mask)


def evaluate(model_name='mse'):
    '''
    Evaluate an already trained model
    :param model_name --> the model weights that we want to evaluate

    :return --> output the evaluated model weights directly to the screen.
    '''
    test_img_list = sorted(glob.glob(f'data/{cell_type}/test/image/*.jpg'))
    test_mask_list = sorted(glob.glob(f'data/{cell_type}/test/mask/*.jpg'))
    if cell_type == 'rbc':
        test_edge_list = sorted(glob.glob(f'data/{cell_type}/test/edge/*.jpg'))
    elif cell_type == 'wbc' or cell_type == 'plt':
        test_edge_list = None
    else:
        print('Invalid blood cell type!\n')
        return

    # initializing the do_unet model
    if model_type == 'do_unet':
        model = do_unet()
    else:
        model = segnet()

    # load weights
    if os.path.exists(f'models/{model_name}.h5'):
        model.load_weights(f'models/{model_name}.h5')
    else:
        train(model_name)

    # load test data
    if cell_type == 'rbc':
        img, mask, edge = data.load_data(test_img_list, test_mask_list, test_edge_list, padding=padding[0])

        img_chips, mask_chips, edge_chips = data.slice(
            img,
            mask,
            edge,
            padding=padding[1],
            input_size=input_shape[0],
            output_size=output_shape[0]
        )
    elif cell_type == 'wbc' or cell_type == 'plt':
        img, mask = data.load_data(test_img_list, test_mask_list, padding=padding[0])

        img_chips, mask_chips = data.slice(
            img,
            mask,
            padding=padding[1],
            input_size=input_shape[0],
            output_size=output_shape[0]
        )

    # print the evaluated accuracies
    if cell_type == 'rbc':
        print(model.evaluate(img_chips, (mask_chips, edge_chips)))
    else:
        print(model.evaluate(img_chips, (mask_chips)))


def threshold(img='edge.png', imgName='Im037_0'):
    '''
    This is the threshold function, which applied an otsu threshold
    to the input image (param: img)
    :param img --> the image we want to threshold

    :return --> saves the output thresholded image under the folder output/<cell_type>/threshold_<img>.png
    '''
    if not os.path.exists(f'{output_directory}/{imgName}/{img}'):
        print('Image does not exist!')
        return

    # substract if img is edge_mask
    if img == 'edge_mask.png':
        mask = cv2.imread(f'{output_directory}/{imgName}/threshold_mask.png')
        edge = cv2.imread(f'{output_directory}/{imgName}/threshold_edge.png')

        # substract mask - edge
        image = mask - edge
    else:
        # getting the input image
        image = cv2.imread(f'{output_directory}/{imgName}/{img}')

        # convert to grayscale and apply otsu's thresholding
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if cell_type == 'wbc':
            threshold, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        else:
            threshold, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

    # save the resulting thresholded image
    plt.imsave(f'{output_directory}/{imgName}/threshold_{img}', image, cmap='gray')
    

def hough_transform(img='edge.png', imgName='Im037_0'):
    '''
    This is the Circle Hough Transform function (CHT), which counts the
    circles from an input image.
    :param img --> the input image that we want to count circles from.

    :return --> saves the output image under the folder output/<cell_type>/hough_transform.png
    '''
    if not os.path.exists(f'{output_directory}/{imgName}/{img}'):
        print('Image does not exist!')
        return

    # getting the input image
    image = cv2.imread(f'{output_directory}/{imgName}/{img}')
    # convert to grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply surface filter
    img = surfaceFilter(img, min_size=2000)

    img = ((img > 0) * 255.).astype(np.uint8)

    # apply hough circles
    if cell_type == 'rbc':
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDist=33, maxRadius=55, minRadius=28, param1=30, param2=20)
    elif cell_type == 'wbc':
        if model_type == 'do_unet':
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDist=51, maxRadius=120, minRadius=48, param1=70, param2=20)
        else:
            circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDist=70, maxRadius=120, minRadius=33, param1=45, param2=13)
    elif cell_type == 'plt':
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1.3, minDist=20, maxRadius=24, minRadius=5, param1=13, param2=11)
    output = img.copy()

    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 0, 255), 2)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 0, 255), -1)
        # save the output image
        plt.imsave(f'{output_directory}/{imgName}/hough_transform.png',
                   np.hstack([img, output]))
        # show the hough_transform results
        print(f'Hough transform: {len(circles)}')
        if len(circles) == None:
            return 0
        else:
            return len(circles)
    else:
        return 0


def component_labeling(img='edge.png', imgName='Im037_0'):
    '''
    This is the Connected Component Labeling (CCL), which labels all the connected objects from an input image
    :param img --> the input image that we want to apply CCL to.

    :return --> saves the output image under the folder output/<cell_type>/component_labeling.png
    '''
    if not os.path.exists(f'{output_directory}/{imgName}/{img}'):
        print('Image does not exist!')
        return

    # getting the input image
    image = cv2.imread(f'{output_directory}/{imgName}/{img}')
    # convert to grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # converting those pixels with values 1-127 to 0 and others to 1
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    # applying cv2.connectedComponents() 
    num_labels, labels = cv2.connectedComponents(img)
    
    # map component labels to hue val, 0-179 is the hue range in OpenCV
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    output = cv2.merge([label_hue, blank_ch, blank_ch])

    # converting cvt to BGR
    output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)

    # set bg label to black
    output[label_hue==0] = 0
    
    # saving image after Component Labeling
    plt.imsave(f'{output_directory}/{imgName}/component_labeling.png',
               np.hstack([image, output]))

    # show number of labels detected
    print(f'Connected component labeling: {num_labels - 1}')
    return num_labels - 1


def distance_transform(img='threshold_edge_mask.png', imgName='Im037_0'):
    '''
    This is the Euclidean Distance Transform function (EDT), which applied the distance transform algorithm to an input image>
    :param img --> the input image that we want to apply EDT to.

    :return --> saves the output image under the folder output/<cell_type>/distance_transform.png
    '''
    if not os.path.exists(f'{output_directory}/{imgName}/{img}'):
        print('Image does not exist!')
        return

    # getting the input image
    image = cv2.imread(f'{output_directory}/{imgName}/{img}')
    # convert to numpy array
    img = np.asarray(image)
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = ndimage.distance_transform_edt(img)

    # saving image after Component Labeling
    plt.imsave(f'{output_directory}/{imgName}/distance_transform.png', img, cmap='gray')


def surfaceFilter(image, min_size = None, max_size = None):
    img = image.copy()
    ret, labels = cv2.connectedComponents(img)
    
    label_codes = np.unique(labels)
    result_image = labels
    
    if 9999  in result_image:
        print("error the image contains the null number 9999")
    
    i = 0
    background_index = 0
    max = 0
    for label in label_codes:
        count = (labels == label).sum()

        #find the background index
        if count > max:
            max = count
            background_index = i

        if min_size is not None and (count < min_size):
            result_image[labels == label] = 9999

        if max_size is not None and (count > max_size):
            result_image[labels == label] = 9999
        i = i + 1
    result_image[result_image == 9999] = label_codes[background_index]
    return result_image


def count(img='threshold_mask.png', imgName='Im037_0'):
    if not os.path.exists(f'{output_directory}/{imgName}/{img}'):
        print('Image does not exist!')
        return

    # getting the input image
    image = cv2.imread(f'{output_directory}/{imgName}/{img}')
    # convert to numpy array
    img = np.asarray(image)
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    if cell_type == 'rbc':
        min_distance = 40
    elif cell_type == 'wbc':
        min_distance = 51
        threshold_abs = 24
    elif cell_type == 'plt':
        min_distance = 52
        img = ndimage.binary_dilation(img)
        threshold_abs = None

    edt = ndimage.distance_transform_edt(img)

    coords = peak_local_max(edt, 
                            indices=True,
                            num_peaks=2000,
                            min_distance=min_distance, 
                            threshold_abs=threshold_abs,
                            exclude_border=False,
                            labels=img)

    # print(coords[:, 1])
    canvas = np.ones(img.shape + (3,), dtype=np.uint8) * 255
    i = 255
    for c in coords:
        o_c = (int(c[1]), int(c[0]))
        cv2.circle(canvas, o_c, 20, (i, 0, 0), -1)
        i = i - 1

    # saving image after counting
    plt.imsave(f'{output_directory}/{imgName}/output.png', canvas, cmap='gray')
    print(f'Euclidean Distance Transform:  {len(coords)}')
    return len(coords)


def accuracy(real, predicted):
    acc = (1 - (np.absolute(int(predicted) - int(real)) / int(real))) * 100
    if real == 0 and predicted == 0:
        return 100
    if acc <= 100 and acc > 0:
        return acc
    elif acc < 0:
        return np.absolute(acc / 100)
    else:
        return 0


def predict_all_idb():
    image_list = sorted(glob.glob(f'data/ALL-IDB1-{cell_type}/*'))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    real_count = []
    with open(f'data/{cell_type}_count.txt', 'r+') as rc:
        file = rc.read().splitlines()
        for line in file:
            real_count += [line.split(' ')[-1]]

    i = 0
    cht_accuracy = []
    ccl_accuracy = []
    edt_accuracy = []
    with open(f'{output_directory}/{cell_type}_results.txt', 'a+') as r:
        r.write('Image Real_Count CHT CCL EDT CHT_Accuracy CCL_Accuracy EDT_Accuracy\n')
        for image in image_list:
            img = image.split('/')[-1].split('.')[0]
            print(f'--------------------------------------------------')
            predict(img)
            threshold('mask.png', img)
            print(f'Image <-- {img} -->')
            print(f'Real Count: {real_count[i]}')
            if cell_type == 'rbc':
                threshold('edge.png', img)
                threshold('edge_mask.png', img)
                distance_transform('threshold_edge_mask.png', img)
                cht_count = hough_transform('edge.png', img)
            else:
                distance_transform('threshold_mask.png', img)
                cht_count = hough_transform('threshold_mask.png', img)

            edt_count = count('threshold_mask.png', img)
            ccl_count = component_labeling('threshold_mask.png', img)
            cht_accuracy += [accuracy(real_count[i], cht_count)]
            ccl_accuracy += [accuracy(real_count[i], ccl_count)]
            edt_accuracy += [accuracy(real_count[i], edt_count)]
            # accuracy = np.mean([cht_accuracy, ccl_accuracy])
            r.write(f'{img} {real_count[i]} {cht_count} {ccl_count} {edt_count} {cht_accuracy[i]} {ccl_accuracy[i]} {edt_accuracy[i]}\n')
            i = i + 1

        r.write(f'CHT Accuracy: {np.mean(cht_accuracy)}\n')
        r.write(f'CCL Accuracy: {np.mean(ccl_accuracy)}\n')
        r.write(f'EDT Accuracy: {np.mean(edt_accuracy)}\n')


if __name__ == '__main__':
    '''
    The main function, which handles all the function call
    (later on, this will dynamically call functions according user input)
    '''
    # train('wbc_segnet', epochs=250)
    # evaluate(model_name='wbc_segnet')
    # image = 'Im079_0'
    # predict(imgName=image)
    # threshold('mask.png', image)

    # if cell_type == 'rbc':
    #     threshold('edge.png', image)
    #     threshold('edge_mask.png', image)
    #     distance_transform('threshold_edge_mask.png', image)
    #     hough_transform('edge.png', image)
    # else:
    #     distance_transform('threshold_mask.png', image)
    #     hough_transform('threshold_mask.png', image)

    # count('threshold_mask.png', image)
    # component_labeling('threshold_mask.png', image)

    predict_all_idb()

