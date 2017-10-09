# visualize.py
# Andrew Kramer

# accepts predicted and ground truth results from camera-pose.py
# visualizes a random sample of the results as vectors comparing
# the prediction to the ground truth orientation change

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Visualize:

	# accepts vectors representing the ground truth relative pose
	# and the predicted relative pose represented by a 3D translation
	# and an orientation quaternion
	def __init__(self, g_truth, pred):
		self.groundtruth = self.__to_euler(np.array(g_truth))
		self.prediction = self.__to_euler(np.array(pred))

	# accepts a 7D vector with indices 0-2 representing X, Y, and Z translation
	# and indices 3-6 representing an orientation quaternion
	# returns a 6D vector with the quaternion converted to a Euler angle
	def __to_euler(self, v):
		r = np.zeros((v.shape[0], v.shape[1] - 1))
		r[:,:3] = v[:,:3]
		r[:,3] = np.arctan2((2 * (v[:,3]*v[:,4]+v[:,5]*v[:,6])), 
							(1-(2*(np.square(v[:,4])+np.square(v[:,5])))))
		r[:,4] = np.arcsin(2 * ((v[:,3]*v[:,5]) - (v[:,6]*v[:,4])))
		r[:,5] = np.arctan2((2 * (v[:,3]*v[:,6]+v[:,4]*v[:,5])), 
							(1-(2*(np.square(v[:,5])+np.square(v[:,6])))))
		return r

	def plot(self, i):
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		ax.set_xlim(-1, 1)
		ax.set_ylim(-1, 1)
		ax.set_zlim(-1, 1)
		#plt.title('Camera Pose Transform')

		x = [0]
		y = [0]
		z = [0]
		u = [1]
		v = [0]
		w = [0]
		c = [1]

		
		
		
		g_roll = self.groundtruth[i, 3]
		g_pitch = self.groundtruth[i, 4]
		g_yaw = self.groundtruth[i, 5]
			
		gi = np.cos(g_yaw) * np.cos(g_pitch) + 1
		gj = np.sin(g_yaw) * np.cos(g_pitch)
		gk = np.sin(g_pitch) 
		
		x.append(self.groundtruth[i, 0])
		y.append(self.groundtruth[i, 1])
		z.append(self.groundtruth[i, 2])
		u.append(gi)
		v.append(gj)
		w.append(gk)
		c.append(0.5)

		p_roll = self.prediction[i, 3]
		p_pitch = self.prediction[i, 4]
		p_yaw = self.prediction[i, 5]

		pi = np.cos(p_yaw) * np.cos(p_pitch) + 1
		pj = np.sin(p_yaw) * np.cos(p_pitch)
		pk = np.sin(p_pitch) 

		x.append(self.prediction[i, 0])
		y.append(self.prediction[i, 1])
		z.append(self.prediction[i, 2])
		u.append(pi)
		v.append(pj)
		w.append(pk)
		c.append(0.1)
		
		cmap = plt.get_cmap()
		ax.quiver(x, y, z, u, v, w, color=cmap(c), length=.5, normalize=True)

		plt.show()
