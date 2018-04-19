import fnmatch
import os
import random

import numpy as np
from skimage.io import imread_collection
from skimage.transform import rotate, rescale

from utils import show_images, save_image

OBJECTS_PATH = '../dataset/preprocessed/others/cropped/*.jpg'
XBOX_PATH = '../dataset/preprocessed/others/xbox.jpg'
PHONE_PATH = '../dataset/preprocessed/others/phone.jpg'
SAVE_PATH = '../dataset/preprocessed/combined/bagpacks/G0001/G0001_'
BAGS_PATH = '../dataset/preprocessed/combined/bagpacks'


def main():
    images = read_images()

    # 1019 to be precise :D
    for i in range(1000):
        generated = generate_image(images)
        generated_name = SAVE_PATH + str(i).zfill(4) + ".png"
        print("Saving " + generated_name)
        save_image(generated_name, generated)
        # show_images([generated])


def generate_image(images):
    background = np.full((1024, 1024), True)

    selected = select_randomly(images)

    for image in selected:
        image = resize_randomly(image)
        image = rotate_randomly(image)
        insert_randomly(background, image)

    return background


def rotate_randomly(image):
    return rotate(image, random.randint(0, 360), resize=True, cval=True).astype(np.bool)


def resize_randomly(image):
    return rescale(image, random.uniform(0.75, 1.1)).astype(np.bool)


def insert_randomly(background, image):
    x = random.randint(0, background.shape[0] - image.shape[0])
    y = random.randint(0, background.shape[1] - image.shape[1])
    inset_at(background, image, x, y)


def inset_at(background, image, x, y):
    background[x:x + image.shape[0], y:y + image.shape[1]] *= image


def select_randomly(images):
    return random.sample(images, random.randint(3, len(images) - 1))


def read_images():
    images = imread_collection(OBJECTS_PATH)
    image_list = []
    for (image, name) in zip(images, images.files):
        image_list.append(image.astype(np.bool))
    return image_list


main()
