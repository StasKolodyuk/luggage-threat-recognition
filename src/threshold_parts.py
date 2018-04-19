from skimage import util
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from skimage.io import imread_collection

from utils import save_image

OBJECTS_PATH = '../dataset/initial/others/cropped/*.jpg'
SINGLE_OBJECT_PATH = '../dataset/initial/others/xbox.jpg'


def main():
    images = imread_collection(OBJECTS_PATH)

    for (image, name) in zip(images, images.files):
        grayscale = rgb2gray(image)
        inverted = util.invert(grayscale)
        threshed = inverted > threshold_otsu(inverted)
        save_image(name.replace("initial", "preprocessed"), threshed)


main()
