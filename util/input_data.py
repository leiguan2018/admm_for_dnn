# coding: utf-8

import cupy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical
import numpy
from scipy.io import loadmat
import os
import logging

class DataSubset(object):
	def __init__(self, xs, ys):
		self.xs = xs
		self.n = xs.shape[0]
		self.ys = ys
		self.batch_start = 0
		self.cur_order = np.random.permutation(self.n)

	def next_batch(self, batch_size, reshuffle_after_pass=True, swapaxes=False):
		if self.n < batch_size:
			raise ValueError('Batch size can be at most the dataset size')
		actual_batch_size = min(batch_size, self.n - self.batch_start)
		if actual_batch_size < batch_size:
			if reshuffle_after_pass:
				self.cur_order = np.random.permutation(self.n)
			self.batch_start = 0
		batch_end = self.batch_start + batch_size
		batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
		batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
		self.batch_start += batch_size
		if swapaxes:
			batch_xs = np.swapaxes(batch_xs, 0, 1)
			batch_ys = np.swapaxes(batch_ys, 0, 1)
		return batch_xs, batch_ys

import scipy
class mnist():

	def __init__(self):
		self.mnist = tf.keras.datasets.mnist		
		(self.x_train, self.y_train), (self.x_test, self.y_test) = self.mnist.load_data()
		#print("origin shape", self.x_train.shape, self.y_train.shape, self.x_test.shape, self.y_test.shape)

		# Uncomment to normalize to (0, 1)
		self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0


		self.x_train = self.x_train.reshape(60000, 28 * 28)
		self.y_train = to_categorical(self.y_train, num_classes=10).reshape(60000, 10)

		self.x_test = self.x_test.reshape(10000, 28 * 28)
		self.y_test = to_categorical(self.y_test, num_classes=10).reshape(10000, 10)

		self.train = DataSubset(self.x_train, self.y_train)
		self.test = DataSubset(self.x_test, self.y_test)

class fashion_mnist():
	def __init__(self):
		self.mnist = tf.keras.datasets.fashion_mnist		
		(self.x_train, self.y_train), (self.x_test, self.y_test) = self.mnist.load_data()

		# Uncomment to normalize to (0, 1)
		self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

		self.x_train = self.x_train.reshape(60000, 28 * 28)
		self.y_train = to_categorical(self.y_train, num_classes=10).reshape(60000, 10)

		self.x_test = self.x_test.reshape(10000, 28 * 28)
		self.y_test = to_categorical(self.y_test, num_classes=10).reshape(10000, 10)
		self.train = DataSubset(self.x_train, self.y_train)
		self.test = DataSubset(self.x_test, self.y_test)


class cifar10():
	def __init__(self):
		self.cifar10 = tf.keras.datasets.cifar10
		(self.x_train, self.y_train), (self.x_test, self.y_test) = self.cifar10.load_data()
		# print("origin shape", self.x_train.shape, self.y_train.shape, self.x_test.shape, self.y_test.shape)

		# Uncomment to normalize to (0, 1)
		self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0


		self.x_train = self.x_train.reshape(50000, 32 * 32 * 3)
		self.y_train = to_categorical(self.y_train, num_classes=10).reshape(50000, 10)

		self.x_test = self.x_test.reshape(10000, 32 * 32 * 3)
		self.y_test = to_categorical(self.y_test, num_classes=10).reshape(10000, 10)

		self.train = DataSubset(self.x_train, self.y_train)
		self.test = DataSubset(self.x_test, self.y_test)


class cifar100():
	def __init__(self):
		self.cifar100 = tf.keras.datasets.cifar100
		(self.x_train, self.y_train), (self.x_test, self.y_test) = self.cifar100.load_data()
		# print("origin shape", self.x_train.shape, self.y_train.shape, self.x_test.shape, self.y_test.shape)

		# Uncomment to normalize to (0, 1)
		self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0


		self.x_train = self.x_train.reshape(50000, 32 * 32 * 3)
		self.y_train = to_categorical(self.y_train, num_classes=100).reshape(50000, 100)

		self.x_test = self.x_test.reshape(10000, 32 * 32 * 3)
		self.y_test = to_categorical(self.y_test, num_classes=100).reshape(10000, 100)

		self.train = DataSubset(self.x_train, self.y_train)
		self.test = DataSubset(self.x_test, self.y_test)


class svhn():
	def __init__(self):
		self.train_data = loadmat('/home/lei/project/admm_dnn/admm_dnn_code/data/train_32x32.mat')
		self.test_data = loadmat('/home/lei/project/admm_dnn/admm_dnn_code/data/test_32x32.mat')

		self.x_train, self.y_train = self.train_data['X'], self.train_data['y']
		self.x_test, self.y_test = self.test_data['X'], self.test_data['y']

		# print("shape0", self.x_train.shape, self.y_train.shape)

		self.x_train = numpy.rollaxis(self.x_train, 3)
		self.x_test = numpy.rollaxis(self.x_test, 3)

		self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

		# print("shape1", self.x_train.shape, self.y_train.shape)

		self.y_train = self.y_train[:, 0]
		self.y_test = self.y_test[:, 0]
		self.y_train[self.y_train == 10] = 0
		self.y_test[self.y_test == 10] = 0

		self.x_train = self.x_train.reshape(73257, 32 * 32 * 3)
		self.y_train = to_categorical(self.y_train, num_classes=10).reshape(73257, 10)

		self.x_test = self.x_test.reshape(26032, 32 * 32 * 3)
		self.y_test = to_categorical(self.y_test, num_classes=10).reshape(26032, 10)

		self.train = DataSubset(self.x_train, self.y_train)
		self.test = DataSubset(self.x_test, self.y_test)

_logger = logging.getLogger(__name__)
class emnist():
	def __init__(self):
		EMNIST_FILENAME = 'emnist-balanced.mat'
		emnist_folder_path = "/home/lei/project/data/matlab/"
		filepath = os.path.join(emnist_folder_path, EMNIST_FILENAME)

		if not (os.path.isdir(emnist_folder_path) and os.path.isfile(filepath)):
			error_msg = ("Folder {} or file {} does not exist "
						 "and download is deactivated").format(
				emnist_folder_path, EMNIST_FILENAME)
			raise FileNotFoundError(error_msg)

		raw_emnist_data = loadmat(filepath)
		_logger.info("Loading train and test data from %s", filepath)
		# load train data
		x_train = raw_emnist_data["dataset"][0][0][0][0][0][0]
		x_train = x_train.reshape((x_train.shape[0], 28, 28), order='A')
		y_train = raw_emnist_data["dataset"][0][0][0][0][0][1]
		# load test data
		x_test = raw_emnist_data["dataset"][0][0][1][0][0][0]
		x_test = x_test.reshape((x_test.shape[0], 28, 28), order='A')
		y_test = raw_emnist_data["dataset"][0][0][1][0][0][1]
		# load mapping from label to chr
		# mapping = raw_emnist_data["dataset"][0][0][2]

		x_train, x_test = x_train / 255.0, x_test / 255.0

		x_train = x_train.reshape(112800, 28 * 28)
		y_train = to_categorical(y_train, num_classes=47).reshape(112800, 47)
		x_test = x_test.reshape(18800, 28 * 28)
		y_test = to_categorical(y_test, num_classes=47).reshape(18800, 47)


		self.train = DataSubset(x_train, y_train)
		self.test = DataSubset(x_test, y_test)