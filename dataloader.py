# camera-pose.py
# Andrew Kramer

# loads data and labels for camera pose estimation as presented in
# https://arxiv.org/pdf/1702.01381.pdf

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

class DataLoader:

	num_images_1 = 49 # number of images in type 1 sets (1-77)
	num_images_2 = 64 # number of images in type 2 sets (82-128)
	input_shape = []
	train_labels = []
	test_labels = []
	train_data = []
	test_data = []
	input_shape = []

	# accepts the name of a directory and the name of a .npy file as strings
	# loads data from the given .npy if it exists, otherwise loads data from
	# raw images and saves it to a .npy file for future runs
	# returns a numpy array representation of
	# the image set in the given directiory
	def __load_images(self, directories, img_rows, img_cols):
		image_set = []
		cwd = os.getcwd() # save current working directory
		for directory_num in directories:
			new_images = []
			os.chdir("/Users/andrew-kramer/Downloads/Cleaned/scan%s" % directory_num) # switch to directory for image files
			if os.path.isfile("path%s.npy" % directory_num):
				new_images = np.load("path%s.npy" % directory_num);
			else:
				for file in glob.glob("*max.png"): # only loads the 'max' image from each view
					img = image.load_img(file, target_size=(img_rows, img_cols))
					img_array = image.img_to_array(img)
					# do some other preprocessing here?
					new_images.append(img_array)
				np.save("path%s.npy" % directory_num, image_set);
			if not np.array(image_set).size:
				image_set = new_images
			else:
				image_set= np.concatenate((image_set, new_images), 0)
		os.chdir(cwd) # switch back to previous working directory

		#preprocess input
		if K.image_data_format() == 'channels_first':
			image_set = image_set.reshape(image_set.shape[0], 3, img_rows, img_cols)
			self.input_shape = (3, img_rows, img_cols)
		else:
			image_set = image_set.reshape(image_set.shape[0], img_rows, img_cols, 3)
			self.input_shape = (img_rows, img_cols, 3)
		image_set = image_set.astype('float32')
		image_set /= 255
		return image_set

	# accepts an array of categoryIDs as a parameter
	# loads ground relative pose data for images in those categories
	def __load_labels(self, categoryIDs):
		labels = []
		data_files = ["data/train_data_mvs.txt", "data/test_data_mvs.txt"]
		for file in data_files:
			f = open(file)
			for line in f:
				if line[0].isdigit():
					sl = line.split()
					nextTuple = (int(sl[0]), int(sl[1]), int(sl[2]),
								 float(sl[3]), float(sl[4]), float(sl[5]),
								 float(sl[6]), float(sl[7]), float(sl[8]), float(sl[9]))
					if nextTuple[2] in categoryIDs:
						labels.append(nextTuple);
			f.close()
		return np.array(labels)

	# accepts array of labels with image identifiers
	# returns shuffled labels split into training and testing sets
	def __organize_labels(self, labels):
		np.random.shuffle(labels)
		num_labels = np.array(labels).shape[0]
		train_index = int(0.8*num_labels)
		train_labels = labels[:train_index,:]
		test_labels = labels[train_index:,:]
		return train_labels, test_labels

	# accepts arrays of training and testing labels, 
	# array of images, and array of category IDs
	# returns arrays of image tuples representing the training and
	# testing datasets
	def __organize_data(self, train_labels, test_labels, images, categoryIDs):
		train_data = []
		test_data = []
		for label in train_labels:
			if label[2] <= 77: # currently can't category IDs above 77
				mult = categoryIDs.index(label[2])
				image_tuple = (images[mult * self.num_images_1], images[mult * self.num_images_1])
				train_data.append(image_tuple)
		for label in test_labels:
			if label[2] <= 77: # currently can't category IDs above 77
				mult = categoryIDs.index(label[2])
				image_tuple = (images[mult * self.num_images_1], images[mult * self.num_images_1])
				test_data.append(image_tuple)
		return np.array(train_data), np.array(test_data)

	# accepts arrays of training and testing labels
	# returs same arrays with image identifying data removed
	# final arrays have form: [relative translation, relative orientation]
	#						  [[x, y, z] [q1, q2, q3, q4]]
	def __clean_labels(self, train_labels, test_labels):
		return train_labels[:,3:], test_labels[:,3:]

	# returns shuffled training and test data consisting of
	# lists of image pairs with indicies [pair_num, image_num, width, height, depth]
	def get_data(self):
		return self.train_data, self.test_data

	# returns shuffled training and test labels with form:
	# [x, y, z, q1, q2, q3, q4]
	def get_labels(self):
		return self.train_labels, self.test_labels

	def get_input_shape(self):
		return self.input_shape

	def __init__(self, categoryIDs, img_rows, img_cols):
		images = self.__load_images(categoryIDs, img_rows, img_cols)
		labels = self.__load_labels(categoryIDs)
		self.train_labels, self.test_labels = self.__organize_labels(labels)
		self.train_data, self.test_data = self.__organize_data(self.train_labels, self.test_labels, images, categoryIDs)
		self.train_labels, self.test_labels = self.__clean_labels(self.train_labels, self.test_labels)
		
