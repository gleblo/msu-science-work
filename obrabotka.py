import cv2
import numpy as np



def consists_of_shit(num):
    shit = {"0","1","-","."}
    set_digs = set(str(num))
    if (set_digs <= shit):
        return True
    else:
        return False


assert consists_of_shit(-0.1) == True
assert (consists_of_shit(0.2) == False)



def remove_shit(im, lbl):

	images = []
	labels = []
	for i in indcs:
	    img = im[i].copy()
        label = lbl[i]

        if len(img)<26  or len(img)>22 or len(img[0]) < 65 or len(img[0]) > 60 or consists_of_shit(label)
            continue

	    img = cv2.resize(img, dsize=(128, 32), interpolation=cv2.INTER_CUBIC)
	    img = np.expand_dims(img , axis = 2)
	    img = img/255.


	    images.append(img)
	    labels.append(label)

	return images, labels


