"""
This script is adapted from:
https://gist.github.com/astanin/626356
and
https://stackoverflow.com/questions/51601272/python-matplotlib-heatmap-colorbar-from-transparent


In this file the main processing happens.

Expectations:
1. All files are in one directory
2. All files are in original size (resizing would make things more difficult)
3. All users used some form of RGB, meaning only #000000 is the edges of the hand images

Hopes:
4. All images are .PNG

"""
import math
import os

import numpy as np

from numpy import average

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from skimage.color import rgba2rgb
from skimage.io import imread, imsave, imread_collection
from skimage.transform import resize
from skimage.util import invert

# ============ Constants ==========
# If True, we save the image, if False, we show the image
save = True

# Locations, relative to the folder this file is in. Have to exist before starting the program
original_image_location = "./images/original/"
participant_image_location = "./images/participants/"
intermediate_proccessing_image_location = "./images/intermediate_processing/"

# Loading of the original image, this is considered as a constant throughout the entire program
original_image_name = os.listdir(original_image_location)
original_image = imread(original_image_location + original_image_name[0])


def main():
    """
    Load the original image and the collection of submitted files.


    :return: None
    """
    participant_answers_collection = imread_collection(participant_image_location + "*", False)

    verify_participant_data(original_image)

    compared_list = compare_images(participant_answers_collection, original_image)

    list_to_heatmap(compared_list, original_image)


def verify_participant_data(original):
    """
    Compares the participant submission with the original image. \n Issues are printed so that the researcher can fix these

    :param original: The original image
    :return: None
    """
    for participant_img_name in os.listdir(participant_image_location):
        image = imread(participant_image_location + participant_img_name)
        if (image.shape == original.shape) and (".png" in participant_img_name):
            pass
        elif (image.shape[2] == original.shape[2]):
            print("Image does not have the same size as original. Color depth is the same."
                  " Issue caused by " + participant_img_name)
        else:
            print(participant_image_location + participant_img_name)
            print("Image does not have same color depth and/or size."
                  "Issue caused by " + participant_img_name)


def compare_images(image_collection, original):
    """

    :param image_collection:
    :param original:
    :return:
    """
    print(len(image_collection), image_collection)
    x = 1
    result_list = []
    for image in image_collection:
        diff = image - original
        diff = to_grayscale(diff)
        diff = simplify(diff)

        # Output intermediate image for some analysis
        imsave(intermediate_proccessing_image_location + 'diff' + str(x) + '.png', diff)
        x += 1
        result_list.append(diff)
    return result_list


def list_to_heatmap(image_list, original):
    """

    :param image_list:
    :param original:
    :return:
    """
    # Creat the heatmap file
    heatmap = np.zeros_like(image_list[0])
    for image in image_list:
        heatmap += image
    draw_plot(heatmap, original)


def draw_plot(heatmap, original):
    """

    :param heatmap:
    :param original:
    :return:
    """
    # === Setup colormap ===
    ncolors = 256
    color_array = plt.get_cmap('viridis')(range(ncolors))
    # change alpha values
    steps = (heatmap.max(initial=0) // 3.5) / ncolors
    print(steps, 1 / steps, "heatmap max ", heatmap.max())
    color_array[0:math.ceil(1 / steps), -1] = np.arange(0.0, 1.0, steps)
    # create a colormap object
    map_object = LinearSegmentedColormap.from_list(name='viridis_alpha', colors=color_array)
    # register this new colormap with matplotlib
    plt.register_cmap(cmap=map_object)
    # === End Setup colormap ===

    # === Remove axis-labels ===
    plt.xticks([1], " ")
    plt.yticks([1], " ")
    # === End Remove axis-labels ===

    # Create the image with heatmap overlayed
    plt.imshow(original, alpha=0.15)
    plt.imshow(heatmap, cmap='viridis_alpha')

    # Some descriptions
    colorbar = plt.colorbar()
    colorbar.ax.set_title("Heatmap strength")
    plt.suptitle("Participant Handshake Presure Heatmap")

    # and save/show
    if save:
        plt.savefig('hand_heatmap.png')
    else:
        plt.show()


def to_grayscale(arr):
    """If arr is a color image (3D array), convert it to grayscale (2D array).


    :param arr:
    :return:
    """

    if len(arr.shape) == 3:
        return average(arr, -1)  # average over the last axis (color channels)
    else:
        return arr


def simplify(arr):
    """

    :param arr:
    :return:
    """
    flipped_arr = invert(arr)
    fixed_arr = flipped_arr + abs(flipped_arr.min())
    divided_arr = fixed_arr // fixed_arr.max()
    flipped_back_arr = invert((divided_arr))
    return flipped_back_arr


if __name__ == "__main__":
    main()
