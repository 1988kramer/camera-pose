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

# accepts the name of a directory and the name of a .npy file as strings
# loads data from the given .npy if it exists, otherwise loads data from
# raw images and saves it to a .npy file for future runs
# returns a numpy array representation of
# the image set in the given directiory
def load_images(directory, file_name):
	image_set = []
	cwd = os.getcwd() # save current working directory
	os.chdir(directory) # switch to directory for image files
	if os.path.isfile(file_name):
		image_set = np.load(file_name);
	else:
		for file in glob.glob("*max.png"): # only loads the 'max' image from each view
			img = image.load_img(file, target_size=input_size)
			img_array = image.img_to_array(img)
			# do some other preprocessing here?
			image_set.append(img_array)
		np.save(file_name, image_set);
	os.chdir(cwd) # switch back to previous working directory
	return image_set

# accepts an array of categoryIDs as a parameter
# loads ground relative pose data for images in those categories
def load_labels(categoryID):
	labels = []
	data_files = ["data/train_data_mvs.txt", "data/test_data_mvs.txt"]
	for file in data_files:
		f = open(file)
		for line in f:
			if line[0].isdigit():
				sl = line.split()
				nextTuple = [int(sl[0]), int(sl[1]), int(sl[2]),
							 float(sl[3]), float(sl[4]), float(sl[5]),
							 float(sl[6]), float(sl[7]), float(sl[8]), float(sl[9])]
				#print(nextTuple)
				if nextTuple[2] == categoryID:
					labels.append(nextTuple);
		f.close()
	return labels

if __name__ == "__main__":
	for i in range(82,129):
		image_set = load_images('/Users/andrew-kramer/Downloads/Cleaned/scan%s' % i, 'path%d.npy' % i)
		print("size of image array: ", np.array(image_set).shape)
	#labels = load_labels(1)
	
	print("size of label array: ", np.array(labels).shape)
