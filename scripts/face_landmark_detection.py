#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   This face detector is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset.
#
#   Also, note that you can train your own models using dlib's machine learning
#   tools. See train_shape_predictor.py to see an example.
#
#   You can get the shape_predictor_68_face_landmarks.dat file from:
#   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.  
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html. 

import sys
import os
import dlib
import glob
from skimage import io

import numpy as np

if len(sys.argv) != 4:
    print(
        "Give the path to the trained shape predictor model as the first "
        "argument and then the directory containing the facial images.\n"
        "For example, if you are in the python_examples folder then "
        "execute this program by running:\n"
        "    ./face_landmark_detection.py shape_predictor_68_face_landmarks.dat ../examples/faces\n"
        "You can download a trained facial shape predictor from:\n"
        "    http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    exit()

predictor_path = sys.argv[1]
faces_folder_path = sys.argv[2]
mouth_folder_path = sys.argv[3]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
#win = dlib.image_window()

try:
    os.makedirs(mouth_folder_path)
except:
    pass

for f in glob.glob(os.path.join(faces_folder_path, "*.png")):
    filename = f.split("/")[-1]
    print("Processing file: {}".format(f))
    img = io.imread(f)

    dets = detector(img, 1)
    for k, d in enumerate(dets):
        shape = predictor(img, d)
    	i = -1
	mouth_points = []
	for part in shape.parts():
		i += 1
		if i < 48:
			continue
		mouth_points.append((part.x,part.y))
	np_mouth_points = np.array(mouth_points)
	mouth_centroid = np.mean(np_mouth_points[:,-2:], axis=0).astype(int)
	#print mouth_centroid
	mouth_crop_image = img[mouth_centroid[1]-25:mouth_centroid[1]+25,mouth_centroid[0]-50:mouth_centroid[0]+50]
	#win.set_image(mouth_crop_image)
	#win.add_overlay(dlib.rectangle(mouth_centroid[0]-60,mouth_centroid[1]-60,mouth_centroid[0]+60,mouth_centroid[1]+60))
	io.imsave(os.path.join(mouth_folder_path, "mouth_"+filename), mouth_crop_image)

    #win.add_overlay(dets)
    #dlib.hit_enter_to_continue()
