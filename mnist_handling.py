import random
import numpy as np
# import cv2
from skimage.transform import resize
import tensorflow as tf


def tf_complicater(arr):
	return tf_noiser(tf_shift(tf_resizer(tf_shaper(arr))))



def tf_shift(arr):
	delta = random.uniform(0,1)
	shift = tf.constant(value = delta, shape=tf.shape(arr))
	ans = tf.scalar_mul(scalar = (1./(1+delta)), x = tf.add(arr,shift))
	return ans	




def tf_resizer(image):
	shape = image.get_shape().as_list()
	width  = shape[1]
	heigth = shape[2]



	scale = np.random.random_sample()/2 + 1
	new_width  = int(width*scale)
	new_heigth = int(heigth*scale)

	new_image = tf.image.resize(image, [new_width, new_heigth])
	# new =  resize(arr, output_shape = (length, new_width, new_heigth, 1))
	# print(new.shape, flush = True)

	return new_image




def tf_shaper(arr):
	inp = tf.reshape(arr, [-1, 28, 28, 1])
	return inp


def tf_placer(arr):
	shape = arr.shape
	length = shape[0]
	width  = shape[1]
	heigth = shape[2]
	# pre = arr.reshape((length, 28, 28, 1))

	new_width  = np.maximum(28, np.random.randint(low = width+1, high = width*2))

	new_heigth = np.maximum(28, np.random.randint(low = heigth+1, high = heigth*2))


	result = np.zeros(shape = (length, new_width, new_heigth, 1))
	pos_x  = np.random.randint(low = 0, high = new_width - width,  size = length)
	pos_y  = np.random.randint(low = 0, high = new_heigth- heigth, size = length)

	for i in range(length):
		# for j in range(0, 28):
		# 	for k in range(0, 28):
		# 		# print(i,j,k, flush=True)
		# 		result[i, j+pos_x, k+pos_y, 0] = pre[i, j, k, 0]

		result[i, pos_x[i]:pos_x[i]+width, pos_y[i]:pos_y[i]+heigth] = arr[i]

	return result



def tf_noiser(arr):
	bias = tf.random.normal(shape=tf.shape(arr),stddev=0.1)
	ans = tf.add(arr, bias)
	return ans







def complicater(arr):
	return noiser(placer(resizer(shaper(arr))))



def resizer(arr):
	shape = arr.shape
	# print(shape)
	length = shape[0]
	width  = shape[1]
	heigth = shape[2]


	scale = np.random.random_sample() + 0.5
	new_width  = int(width*scale)
	new_heigth = int(heigth*scale)
	new =  resize(arr, output_shape = (length, new_width, new_heigth, 1))
	# print(new.shape, flush = True)

	return new




def shaper(arr):
	length = len(arr)
	return arr.reshape((length, 28, 28, 1))


def placer(arr):
	shape = arr.shape
	length = shape[0]
	width  = shape[1]
	heigth = shape[2]
	# pre = arr.reshape((length, 28, 28, 1))

	new_width  = np.maximum(28, np.random.randint(low = width+1, high = width*2))

	new_heigth = np.maximum(28, np.random.randint(low = heigth+1, high = heigth*2))


	result = np.zeros(shape = (length, new_width, new_heigth, 1))
	pos_x  = np.random.randint(low = 0, high = new_width - width,  size = length)
	pos_y  = np.random.randint(low = 0, high = new_heigth- heigth, size = length)

	for i in range(length):
		# for j in range(0, 28):
		# 	for k in range(0, 28):
		# 		# print(i,j,k, flush=True)
		# 		result[i, j+pos_x, k+pos_y, 0] = pre[i, j, k, 0]
		result[i, pos_x[i]:pos_x[i]+width, pos_y[i]:pos_y[i]+heigth] = arr[i]

	return result



def noiser(arr):
	shape = arr.shape
	length = shape[0]
	width  = shape[1]
	heigth = shape[2]

	bias = 0.1 * np.random.randn(length, width, heigth, 1)
	ans = arr + bias
	return ans
