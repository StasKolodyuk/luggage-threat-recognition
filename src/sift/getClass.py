#!/usr/local/bin/python2.7

import argparse as ap
import cv2
import imutils
import numpy as np
import os

from scipy._lib.six import xrange
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from scipy.cluster.vq import *

# Load the classifier, class names, scaler, number of clusters and vocabulary
clf, classes_names, stdSlr, k, voc = joblib.load("bof.pkl")

# Get the path of the testing set
parser = ap.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-t", "--testingSet", help="Path to testing Set")
group.add_argument("-i", "--image", help="Path to image")
parser.add_argument('-v', "--visualize", action='store_true')
args = vars(parser.parse_args())

# Get the path of the testing image(s) and store them in a list
image_paths = []
if args["testingSet"]:
    test_path = args["testingSet"]
    try:
        testing_names = os.listdir(test_path)
    except OSError:
        print
        "No such directory {}\nCheck if the file exists".format(test_path)
        exit()
    for testing_name in testing_names:
        dir = os.path.join(test_path, testing_name)
        class_path = imutils.imlist(dir)
        image_paths += class_path
else:
    image_paths = [args["image"]]

# Create feature extraction and keypoint detector objects
fea_det = cv2.xfeatures2d.SIFT_create()

# List where all the descriptors are stored
des_list = []

i = 0
for image_path in image_paths:
    im = cv2.imread(image_path)
    if im == None:
        print
        "No such file {}\nCheck if the file exists".format(image_path)
        exit()
    kpts, des = fea_det.detectAndCompute(im, None)
    des_list.append((image_path, des))
    i += 1
    print("Processed ", image_path, i)

print("Collecting descriptors...")
# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[0:]:
    descriptors = np.vstack((descriptors, descriptor))
print("Descriptors collected...")

print("BOW...")
test_features = np.zeros((len(image_paths), k), "float32")
for i in xrange(len(image_paths)):
    words, distance = vq(des_list[i][1], voc)
    for w in words:
        test_features[i][w] += 1
print("BOW completed...")

print("Tf-Idf...")
# Perform Tf-Idf vectorization
nbr_occurences = np.sum((test_features > 0) * 1, axis=0)
idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)), 'float32')
print("Tf-Idf completed...")

print("Scaling...")
# Scale the features
test_features = stdSlr.transform(test_features)
print("Scaling completed...")

print("Predicting...")
# Perform the predictions
predictions = [classes_names[i] for i in clf.predict(test_features)]
print("Predicting completed...")

matches = 0
overall = 0
for image_path, prediction in zip(image_paths, predictions):
    print("Prediction: ", image_path, prediction)
    overall += 1
    if prediction in image_path:
        matches += 1

print("Success Rate: ", matches/overall)

# Visualize the results, if "visualize" flag set to true by the user
if args["visualize"]:
    for image_path, prediction in zip(image_paths, predictions):
        image = cv2.imread(image_path)
        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        pt = (0, 3 * image.shape[0] // 4)
        cv2.putText(image, prediction, pt, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 2, [0, 255, 0], 2)
        cv2.imshow("Image", image)
        cv2.waitKey(3000)