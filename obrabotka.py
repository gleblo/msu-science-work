
def cut_long_tails(im, lbl)
	images = []
	labels = []
	indcs = [ i for i in range(len(im)) if len(im[i])<26  and len(im[i])>22 and len(im[i][0]) < 65 and len(im[i][0]) > 60 ]
	for i in indcs:
	    img = im[i].copy()

	    img = cv2.resize(img, dsize=(128, 32), interpolation=cv2.INTER_CUBIC)
	    img = np.expand_dims(img , axis = 2)

	    # Normalize each image
	    img = img/255.

	    images.append(img)
	    labels.append(lbl[i])

	return images, labels