import matplotlib.pyplot as plt
from skimage.filters import threshold_minimum, threshold_local, try_all_threshold
from skimage.io import imread_collection

from utils import save_image

BAGPACKS_PATH = '../dataset/initial/combined/bagpacks/*/*.png'
SINGLE_BAGPACK = '../dataset/initial/combined/bagpacks/B0046/B0046_0002.png'


def main():
    images = imread_collection(BAGPACKS_PATH)

    for (image, name) in zip(images, images.files):
        threshed_image = minimum_threshold(image)
        save_image(name.replace("initial", "preprocessed"), threshed_image)


def minimum_threshold(img):
    return img > threshold_minimum(img)


# didn't work
def adaptive_threshold(img):
    block_size = 51
    adaptive_thresh = threshold_local(img, block_size, offset=20)
    binary_adaptive = img > adaptive_thresh

    plt.imshow(binary_adaptive)
    plt.gray()
    plt.show()


# try all thresholds
def all_thresholds(img):
    try_all_threshold(img, figsize=(10, 8), verbose=False)
    plt.show()


main()
