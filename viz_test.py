# viz_test.py
# Andrew Kramer

# testing program for visualize.py

import numpy as np
from visualize import Visualize
import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='results visualizer options')
	parser.add_argument('--index', type=int, required=True, 
						help='index of image pair to use')
	parser.add_argument('--labels', type=str, default='labels.txt',
						help='file name for labels')
	parser.add_argument('--pred', type=str, default='pred.txt',
						help='file name for predictions')
	args = parser.parse_args()

	pred = np.loadtxt(args.pred, delimiter=' ')
	labels = np.loadtxt(args.labels, delimiter=' ')

	viz = Visualize(labels, pred)
	viz.plot(args.index)