# visualize.py
# Andrew Kramer

# accepts predicted and ground truth results from camera-pose.py
# visualizes a random sample of the results as vectors comparing
# the prediction to the ground truth orientation change

import numpy as np
import matplatlib.pyplot as pyplot
from mpl_toolkits.mplot3d import Axes3D

class Visualize:

	# accepts vectors representing the ground truth relative pose
	# and the predicted relative pose represented by a 3D translation
	# and an orientation quaternion
	def __init__(self, g_truth, pred):
		self.groundtruth = to_euler(np.array(g_truth))
		self.prediction = to_euler(np.array(pred))

	# accepts a 7D vector with indices 0-2 representing X, Y, and Z translation
	# and indices 3-6 representing an orientation quaternion
	# returns a 6D vector with the quaternion converted to a Euler angle
	def to_euler(self, vec):
		result = zeros(vec.shape[0], vec.shape[1] - 1)
		for v, r in zip(vec, result):
			r[:3] = v[:3]
			r[3] = np.atan2((2 * (v[3]*v[4]+v[5]*v[6])), (1-(2*(v[4]^2+v[5]^2))))
			r[4] = np.arcsin(2 * (v[3]*v[5]) - (v[6]*v[4]))
			r[5] = np.atan2((2 * (v[3]*v[6]+v[4]*v[5])), (1-(2*(v[5]^2+v[6]^2))))
		return result