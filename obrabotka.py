import cv2
import numpy as np



def consists_of_shit(num):
	shit = {"0","1","-","."}
	set_digs = set(str(num))
	if (set_digs <= shit):
		return True
	else:
		return False


# assert consists_of_shit(-0.1) == True
# assert (consists_of_shit(0.2) == False)



def remove_shit(im, lbl):

	images = []
	labels = []
	for i in range(len(im)):
		img = im[i].copy()
		label = lbl[i]

		if not( len(img)<26 and len(img)>22 and len(img[0]) < 65 and len(img[0]) > 60 and (not consists_of_shit(label)) ):
			continue

		img = cv2.resize(img, dsize=(128, 32), interpolation=cv2.INTER_CUBIC)
		img = np.expand_dims(img , axis = 2)
		img = img/255.


		images.append(img)
		labels.append(label)

	return images, labels




def to_char(num):
    return ("%.1f" % (num))


def remove_dot(txt):
    return txt[:-2] + txt[-1:]



def reparo(lbl):
    return [remove_dot(to_char(lb)) for lb in lbl]



def max_len(lbl):
    return max([len(lb) for lb in lbl])


def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, char in enumerate(txt):
        try:
            dig_lst.append(char_list.index(char))
        except:
            print(char)
        
    return dig_lst

def get_enumed(lbl):
    return [encode_to_labels(lb) for lb in lbl]



# def get_max_len(lbl)
 



def img_prep(imd):
    img = cv2.resize(imd, dsize=(128, 32), interpolation=cv2.INTER_CUBIC)
    img = np.expand_dims(img , axis = 2)

    # Normalize each image
    img = img/255.
    return img

def imgs_prep(im):
    return [img_prep(img) for img in im]

