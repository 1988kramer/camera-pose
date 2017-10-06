# viz_test.py
# Andrew Kramer

# testing program for visualize.py

import numpy as np
from visualize import Visualize

if __name__ == "__main__":
	pred = np.loadtxt('pred.txt', delimiter=' ')
	labels = np.loadtxt('labels.txt', delimiter=' ')

	viz = Visualize(labels, pred)
	viz.plot(3)