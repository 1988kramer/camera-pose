# camera-pose.py
# Andrew Kramer

# regresses the relative camera pose between two images using the method 
# presented in https://arxiv.org/pdf/1702.01381.pdf

import numpy
import tensorflow
import gzip
import sys
import pickle
import keras
import math

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Input, Lambda
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K
from data-loader import DataLoader

beta = 10
epochs = 1

def custom_objective(y_true, y_pred):
	error = y_pred - y_true;
	transMag = np.sqrt(error[0]^2 + error[1]^2 + error[2]^2)
	orientMag = np.sqrt(error[3]^2 + error[4]^2 + error[5]^2 + error[6]^2)
	return transMag + (beta * orientMag)

def dot_product(v1, v2):
	return sum((a*b) for a,b in zip(v1,v2))

def length(v):
	return math.sqrt(dot_product(v,v))

def compute_mean_error(y_true, y_pred):
	trans_error = 0
	orient_error = 0
	for i in range(0,y_true.shape[0]):
		trans_error += math.acos(dot_product(y_true[i,:3], y_pred[i,:3])/
								 (length(y_true[i,:3]) * length(y_pred[i,:3])))
		orient_error += math.acos(dot_product(y_true[i,3:], y_pred[i,3:])/
								  (length(y_true[i,3:]) * lenght(y_pred[i,3:])))
	mean_trans = trans_error / y_true.shape[0]
	mean_orient = orient_error / y_true.shape[0]
	return mean_trans, mean_orient


def create_conv_branch(input_shape):
	model = Sequential()
	model.add(Conv2D(96, kernel_size=(11,11),
					 stride=4, padding='valid',
					 activation='relu',
					 input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2,2), strides=2))
	model.add(Conv2D(256, kernel_size=(5,5),
					 stride=1, padding='same',
					 activation='relu'))
	model.add(MaxPooling2D(pool_size(3,3), strides=1))
	model.add(Conv2D(384, kernel_size=(3,3),
					 stride=1, padding='same',
					 activation='relu'))
	model.add(Conv2D(384, kernel_size=(3,3),
					 stride=1, padding='same',
					 activation='relu'))
	model.add(Conv2D(256, kernel_size=(3,3),
					 stride=1, padding='same',
					 activation='relu'))
	# replace with SPP if possible
	model.add(MaxPooling2D(pool_size=(3,3), strides=2))
	return model

if __name__ == "__main__":

	img_rows, img_cols = 227, 227
	category_IDs = (1,2) # category IDs from which to pull test and training data

	# load training and testing data:
	loader = DataLoader(category_IDs, img_rows, img_cols)
	train_data, test_data = loader.get_data()
	train_labels, test_labels = loader.get_labels()
	input_shape = loader.get_input_shape()

	# define structure of convolutional branches
	conv_branch = create_conv_branch(input_shape)
	branch_a = Input(shape=input_shape)
	branch_b = Input(shape=input_shape)

	processed_a = conv_branch(branch_a)
	processed_b = conv_branch(branch_b)

	# compute distance between outputs of the CNN branches
	# not sure if euclidean distance is right here
	# merging or concatenating inputs may be more accurate
	#distance = Lambda(euclidean_distance, 
	#				  output_shape = eucl_dist_output_shape)([processed_a, processed_b])
	regression = keras.layers.concatenate([processed_a, processed_b])
	regression.add(Flatten()) # may not be necessary
	regression.add(Dense(7, kernel_initializer='normal'))
	model = Model(inputs=[branch_a, branch_b], outputs=[regression])

	model.compile(loss=custom_objective, 
				  optimizer=keras.optimizers.Adam(lr=.0001, decay=.00001),
				  metrics=['accuracy'])

	model.fit([train_data[:,0], train_data[:,1]], train_labels,
			  batch_size=128,
			  epochs = epochs,
			  validation_split=0.1,
			  shuffle=true)

	pred = model.predict(train_data[:,0], train_data[:,1])
	train_trans, train_orient = compute_accuracy(pred, train_labels)
	pred = model.predict(test_data[:,0], test_data[:,1])
	test_trans, test_mean = compute_accuracy(pred, test_labels)

	print('* Mean translation error on training set: %0.2f%%' % (100 * train_trans))
	print('* Mean orientation error on training set: %0.2f%%' % (100 * train_orient))
	print('*     Mean translation error on test set: %0.2f%%' % (100 * test_trans))
	print('*     Mean orientation error on test set: %0.2f%%' % (100 * test_orient))
