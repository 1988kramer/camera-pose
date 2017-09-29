# camera-pose.py
# Andrew Kramer

# regresses the relative camera pose between two images using the method 
# presented in https://arxiv.org/pdf/1702.01381.pdf

import numpy as np
import tensorflow
import keras
import glob, os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing import image
from keras import backend as K

input_size = (227, 227)

# accepts the name of a directory as a string
# returns a numpy array representation of 
# the image set in the given directiory
def load_images(directory):
	image_set = []
	os.chdir(directory)
	for file in glob.glob("*.png"):
		img = image.load_img(file, target_size=input_size)
		img_array = image.img_to_array(img)
		# do some other preprocessing here?
		image_set.append(img_array)
	return image_set

if __name__ == "__main__":
	image_set = load_images('/Users/andrew-kramer/Downloads/Cleaned/scan1')
	print(np.array(image_set).shape)
