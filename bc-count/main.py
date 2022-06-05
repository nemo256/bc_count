import glob
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import ndimage

# custom imports
from config import *
import data
from model import do_unet, get_callbacks


def generate_train_dataset(img_list, mask_list, edge_list=None):
    if cell_type == 'red':
        img, mask, edge = data.load_data(img_list, mask_list, edge_list)
    elif cell_type == 'white':
        img, mask = data.load_data(img_list, mask_list, edge_list)
        edge = None

    if cell_type == 'red':
        def train_gen():
            return data.train_generator(img, mask, edge,
                                        padding=padding[0],
                                        input_size=input_shape[0],
                                        output_size=output_shape[0])
    elif cell_type == 'white':
        def train_gen():
            return data.train_generator(img, mask,
                                        padding=padding[0],
                                        input_size=input_shape[0],
                                        output_size=output_shape[0])

    # load train dataset to tensorflow for training
    if cell_type == 'red':
        return tf.data.Dataset.from_generator(
            train_gen,
            (tf.float64, ((tf.float64), (tf.float64))),
            (input_shape, (output_shape, output_shape))
        )
    elif cell_type == 'white':
        return tf.data.Dataset.from_generator(
            train_gen,
            (tf.float64, (tf.float64)),
            (input_shape, (output_shape))
        )


def generate_test_dataset(img_list, mask_list, edge_list=None):
    if cell_type == 'red':
        img, mask, edge = data.load_data(img_list, mask_list, edge_list)
    elif cell_type == 'white':
        img, mask = data.load_data(img_list, mask_list)

    if cell_type == 'red':
        img_chips, mask_chips, edge_chips = data.test_chips(
            img,
            mask,
            edge,
            padding=padding[1],
            input_size=input_shape[0],
            output_size=output_shape[0]
        )
    elif cell_type == 'white':
        img_chips, mask_chips = data.test_chips(
            img,
            mask,
            padding=padding[1],
            input_size=input_shape[0],
            output_size=output_shape[0]
        )

    # load test dataset to tensorflow for training
    if cell_type == 'red':
        return tf.data.Dataset.from_tensor_slices(
            (img_chips, (mask_chips, edge_chips))
        )
    elif cell_type == 'white':
        return tf.data.Dataset.from_tensor_slices(
            (img_chips, (mask_chips))
        )


def train(model_name='mse', epochs=100):
    # globing appropriate images, their masks and their edges
    if cell_type == 'red':
        train_img_list = sorted(glob.glob('data/rbc/train/image/*.jpg'))
        test_img_list = sorted(glob.glob('data/rbc/test/image/*.jpg'))
        train_mask_list = sorted(glob.glob('data/rbc/train/mask/*.jpg'))
        test_mask_list = sorted(glob.glob('data/rbc/test/mask/*.jpg'))
        train_edge_list = sorted(glob.glob('data/rbc/train/edge/*.jpg'))
        test_edge_list = sorted(glob.glob('data/rbc/test/edge/*.jpg'))
    elif cell_type == 'white':
        train_img_list = sorted(glob.glob('data/wbc/train/image/*.jpg'))
        test_img_list = sorted(glob.glob('data/wbc/test/image/*.jpg'))
        train_mask_list = sorted(glob.glob('data/wbc/train/mask/*.jpg'))
        test_mask_list = sorted(glob.glob('data/wbc/test/mask/*.jpg'))
        train_edge_list = None
        test_edge_list = None
    else:
        train_mask_list = None
        test_mask_list = None
        return False

    # loading train dataset and test datasets
    if cell_type == 'red':
        train_dataset = generate_train_dataset(
            train_img_list,
            train_mask_list,
            train_edge_list,
        )
        test_dataset = generate_test_dataset(
            test_img_list,
            test_mask_list,
            test_edge_list,
        )
    elif cell_type == 'white':
        train_dataset = generate_train_dataset(
            train_img_list,
            train_mask_list,
        )
        test_dataset = generate_test_dataset(
            test_img_list,
            test_mask_list,
        )

    # # initializing the do_unet model
    # model = do_unet()

    # # create models directory if it does not exist
    # if not os.path.exists('models/'):
    #     os.makedirs('models/')

    # # Check for existing weights
    # if os.path.exists(f'models/{model_name}.h5'):
    #     model.load_weights(f'models/{model_name}.h5')

    # # fitting the model
    # history = model.fit(
    #     train_dataset.batch(8),
    #     validation_data=test_dataset.batch(8),
    #     epochs=epochs,
    #     steps_per_epoch=125,
    #     max_queue_size=16,
    #     use_multiprocessing=True,
    #     workers=8,
    #     verbose=1,
    #     callbacks=get_callbacks(model_name)
    # )

    # # save the history
    # np.save(f'models/{model_name}_history.npy', history.history)


def normalize(img):
    '''
    Normalizes an image
    '''
    return np.array((img - np.min(img)) / (np.max(img) - np.min(img)))


# extract number of image chips for an image
def get_sizes(img,
              padding=padding[1],
              input=input_shape[0],
              output=output_shape[0]):
    '''
    Get full image sizes (x, y) to rebuilt the full image output
    '''
    offset = padding + (output / 2)
    return [(len(np.arange(offset, img[0].shape[0] - input / 2, output)), len(np.arange(offset, img[0].shape[1] - input / 2, output)))]


# reshape numpy arrays
def reshape(img,
            size_x,
            size_y):
    '''
    Reshape the full image output using the original sizes (x, y)
    '''
    return img.reshape(size_x, size_y, output_shape[0], output_shape[0], 1)


# concatenate images
def concat(imgs):
    '''
    Concatenate all the output image chips to rebuild the full image
    '''
    return cv2.vconcat([cv2.hconcat(im_list) for im_list in imgs[:,:,:,:]])


# denoise an image
def denoise(img):
    '''
    Remove noise from an image
    '''
    # read the image
    img = cv2.imread(img)
    # return the denoised image
    return cv2.fastNlMeansDenoising(img, 23, 23, 7, 21)


# predict (segment) image and save a sample output
def predict(img='Im037_0',
            model_name='mse'):
    '''
    Predict (segment) blood cell images using the trained model (do_unet)
    '''
    if cell_type == 'red':
        test_img = sorted(glob.glob(f'data/rbc/test/image/{img}.jpg'))
        test_mask = sorted(glob.glob(f'data/rbc/test/mask/{img}_R.jpg'))
        test_edge = sorted(glob.glob(f'data/rbc/test/edge/{img}_RE.jpg'))
    elif cell_type == 'white':
        test_img = sorted(glob.glob(f'data/wbc/test/image/{img}.jpg'))
        test_mask = sorted(glob.glob(f'data/wbc/test/mask/{img}_W.jpg'))
        test_edge = None
    else:
        return False

    # initialize do_unet
    model = do_unet()

    # Check for existing weights
    if os.path.exists(f'models/{model_name}.h5'):
        model.load_weights(f'models/{model_name}.h5')

    # load test data
    if cell_type == 'red':
        img, mask, edge = data.load_data(test_img, test_mask, test_edge, padding=padding[0])

        img_chips, mask_chips, edge_chips = data.test_chips(
            img,
            mask,
            edge,
            padding=padding[1],
            input_size=input_shape[0],
            output_size=output_shape[0]
        )
    elif cell_type == 'white':
        img, mask = data.load_data(test_img, test_mask, padding=padding[0])

        img_chips, mask_chips = data.test_chips(
            img,
            mask,
            padding=padding[1],
            input_size=input_shape[0],
            output_size=output_shape[0]
        )

    # segment all image chips
    if cell_type == 'red':
        output = model.predict(img_chips)
        new_mask_chips = np.array(output[0])
        new_edge_chips = np.array(output[1])
    elif cell_type == 'white':
        output = model.predict(img_chips)
        new_mask_chips = np.array(output)

    # get image dimensions
    dimensions = [get_sizes(img)[0][0], get_sizes(img)[0][1]]

    # reshape chips arrays to be concatenated
    if cell_type == 'red':
        new_mask_chips = reshape(new_mask_chips, dimensions[0], dimensions[1])
        new_edge_chips = reshape(new_edge_chips, dimensions[0], dimensions[1])
    elif cell_type == 'white':
        new_mask_chips = reshape(new_mask_chips, dimensions[0], dimensions[1])

    # get rid of none necessary dimension
    if cell_type == 'red':
        new_mask_chips = np.squeeze(new_mask_chips)
        new_edge_chips = np.squeeze(new_edge_chips)
    elif cell_type == 'white':
        new_mask_chips = np.squeeze(new_mask_chips)

    # concatenate chips into a single image (mask and edge)
    if cell_type == 'red':
        new_mask = concat(new_mask_chips)
        new_edge = concat(new_edge_chips)
    elif cell_type == 'white':
        new_mask = concat(new_mask_chips)
    
    # create output directory if it does not exist
    if not os.path.exists('output/'):
        os.makedirs('output/')

    # save predicted mask and edge
    if cell_type == 'red':
        plt.imsave('output/mask.png', new_mask)
        plt.imsave('output/edge.png', new_edge)
        plt.imsave('output/edge_mask.png', new_mask - new_edge)
    elif cell_type == 'white':
        plt.imsave('output/mask.png', new_mask)

    # organize results into one figure
    if cell_type == 'red':
        fig = plt.figure(figsize=(25, 12), dpi=80)
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        ax  = fig.add_subplot(2, 3, 1)
        ax.set_title('Test image')
        ax.imshow(np.array(img)[0,:,:,:])
        ax  = fig.add_subplot(2, 3, 2)
        ax.set_title('Test mask')
        ax.imshow(np.array(mask)[0,:,:])
        ax  = fig.add_subplot(2, 3, 3)
        ax.set_title('Test edge')
        ax.imshow(np.array(edge)[0,:,:])
        ax  = fig.add_subplot(2, 3, 5)
        ax.set_title('Predicted mask')
        ax.imshow(new_mask)
        ax  = fig.add_subplot(2, 3, 6)
        ax.set_title('Predicted edge')
        ax.imshow(new_edge)
    elif cell_type == 'white':
        fig = plt.figure(figsize=(25, 12), dpi=80)
        fig.subplots_adjust(hspace=0.1, wspace=0.1)
        ax  = fig.add_subplot(2, 2, 1)
        ax.set_title('Test image')
        ax.imshow(np.array(img)[0,:,:,:])
        ax  = fig.add_subplot(2, 2, 2)
        ax.set_title('Test mask')
        ax.imshow(np.array(mask)[0,:,:])
        ax  = fig.add_subplot(2, 2, 4)
        ax.set_title('Predicted mask')
        ax.imshow(new_mask)

    # save the figure as a sample output
    plt.savefig('sample.png')


# evaluate model accuracies (mask accuracy and edge accuracy)
def evaluate(model_name='mse'):
    '''
    Evaluate an already trained model
    '''
    if cell_type == 'red':
        test_img_list = sorted(glob.glob('data/rbc/test/image/*.jpg'))
        test_mask_list = sorted(glob.glob('data/rbc/test/mask/*.jpg'))
        test_edge_list = sorted(glob.glob('data/rbc/test/edge/*.jpg'))
    elif cell_type == 'white':
        test_img_list = sorted(glob.glob('data/wbc/test/image/*.jpg'))
        test_mask_list = sorted(glob.glob('data/wbc/test/mask/*.jpg'))
        test_edge_list = None
    else:
        return False

    # initialize do_unet
    model = do_unet()

    # load weights
    if os.path.exists(f'models/{model_name}.h5'):
        model.load_weights(f'models/{model_name}.h5')
    else:
        train(model_name)

    # load test data
    if cell_type == 'red':
        img, mask, edge = data.load_data(test_img, test_mask, test_edge, padding=padding[0])

        img_chips, mask_chips, edge_chips = data.test_chips(
            img,
            mask,
            edge,
            padding=padding[1],
            input_size=input_shape[0],
            output_size=output_shape[0]
        )
    elif cell_type == 'white':
        img, mask = data.load_data(test_img, test_mask, padding=padding[0])

        img_chips, mask_chips = data.test_chips(
            img,
            mask,
            padding=padding[1],
            input_size=input_shape[0],
            output_size=output_shape[0]
        )

    # print the evaluated accuracies
    if cell_type == 'red':
        print(model.evaluate(img_chips, (mask_chips, edge_chips)))
    else:
        print(model.evaluate(img_chips, (mask_chips)))


# threshold an image using otsu's threshold
def threshold(img='edge.png'):
    if not os.path.exists(f'output/{img}'):
        print('Image does not exist!')
        return

    # substract if img is edge_mask
    if img == 'edge_mask.png':
        mask = cv2.imread(f'output/threshold_mask.png')
        edge = cv2.imread(f'output/threshold_edge.png')

        # substract mask - edge
        image = mask - edge
    else:
        # getting the input image
        image = cv2.imread(f'output/{img}')

        # convert to grayscale and apply otsu's thresholding
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        otsu_threshold, image = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU,)

    # save the resulting thresholded image
    plt.imsave(f'output/threshold_{img}', image, cmap='gray')
    

# count how many cells from the predicted edges
def hough_transform(img='edge.png'):
    if not os.path.exists(f'output/{img}'):
        print('Image does not exist!')
        return

    # getting the input image
    image = cv2.imread(f'output/{img}')
    # convert to grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # apply hough circles
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, minDist=33, maxRadius=55, minRadius=28, param1=30, param2=20)
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
        plt.imsave('output/hough_transform.png',
                   np.hstack([img, output]))

    # show the hough_transform results
    print(f'Hough transform: {len(circles)}')


# count how many cells from the predicted edges
def component_labeling(img='edge.png'):
    if not os.path.exists(f'output/{img}'):
        print('Image does not exist!')
        return

    # getting the input image
    image = cv2.imread(f'output/{img}')
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
    plt.imsave('output/component_labeling.png',
               np.hstack([image, output]))

    # show number of labels detected
    print(f'Connected component labeling: {num_labels}')


# get a minimal of each cell to help with the counting
def distance_transform(img='threshold_edge_mask.png'):
    if not os.path.exists(f'output/{img}'):
        print('Image does not exist!')
        return

    # getting the input image
    image = cv2.imread(f'output/{img}')
    # convert to numpy array
    img = np.asarray(image)
    # convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = ndimage.distance_transform_edt(img)
    img = ndimage.binary_dilation(img)

    # saving image after Component Labeling
    plt.imsave('output/distance_transform.png', img, cmap='gray')



if __name__ == '__main__':
    train('wbc')
    # evaluate(model_name='quadtree_test')
    # predict(model_name='mse')
    # threshold('mask.png')
    # threshold('edge.png')
    # threshold('edge_mask.png')
    # distance_transform('threshold_edge_mask.png')
    # hough_transform('edge.png')
    # component_labeling('distance_transform.png')
