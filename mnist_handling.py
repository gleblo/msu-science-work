import random
import numpy as np
import cv2

def complicater(arr):
	length = len(arr)
	pre = arr.reshape((length, 28, 28, 1))

	width  = random.randint(28,56)
	pos_x = random.randint(0,width-28)

	heigth = random.randint(28,56)
	pos_y  = random.randint(0, heigth-28)


	result = np.zeros(shape = (length, width, heigth, 1))

	for i in range(length):
		for j in range(0, 28):
			for k in range(0, 28):
				# print(i,j,k, flush=True)
				result[i, j+pos_x, k+pos_y, 0] = pre[i, j, k, 0]

	
	bias = np.zeros(shape = (length, width, heigth, 1))
	cv2.randn(bias,(0),(0.1))
	ans = cv2.add(result, bias)
	

	return ans
