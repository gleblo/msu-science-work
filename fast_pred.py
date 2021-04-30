import tensorflow.compat.v1 as tf
from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
import fitz
import shutil
import os

# Loadig toloka data
def load_data_toloka1():
    char_list = list('-0123456789')
    import os
    import numpy as np

    import os
    import fnmatch
    import cv2
    import numpy as np
    import string
    import time
    from keras.preprocessing.sequence import pad_sequences

    from keras.utils import to_categorical
    from keras.callbacks import ModelCheckpoint
    
    def encode_to_labels(txt):
        # encoding each output word into digits
        dig_lst = []
        for index, char in enumerate(txt):
            try:
                dig_lst.append(char_list.index(char))
            except:
                print(char)
            
        return dig_lst



    import pandas as pd
    paths = pd.read_csv('/content/drive/My Drive/15k toloka/checked_images.csv')#,keep_default_na=False)
    paths = paths.dropna()
    # paths.fillna('')




    paths = pd.DataFrame(columns = ["file", "label"])
    for x in os.listdir("/content/drive/My Drive/15k toloka/hehe/kek/lol/"):
        paths = paths.append(pd.Series({"file":x, "label":x.split("_")[0]}), ignore_index= True)
        
    paths = paths.dropna()



    # paths
    paths = paths.sample(frac=1).reset_index(drop=True)
    # paths.sample(frac=1)


    path = '/content/drive/My Drive/15k toloka/hehe/kek/lol'
    
    
    # lists for training dataset
    training_img = []
    training_txt = []
    train_input_length = []
    train_label_length = []
    orig_txt = []
    
    #lists for validation dataset
    valid_img = []
    valid_txt = []
    valid_input_length = []
    valid_label_length = []
    valid_orig_txt = []
    
    max_label_len = 0
    read_again = False


    if read_again == False:
        files = np.load("./drive/My Drive/15k toloka/dat11 без знака.npz",allow_pickle=True)

        valid_orig_txt = files["arr_0"]
        valid_label_length = files["arr_1"]
        valid_input_length = files["arr_2"]
        valid_img = files["arr_3"]
        valid_txt = files["arr_4"]
        orig_txt = files["arr_5"]
        train_label_length = files["arr_6"]
        train_input_length = files["arr_7"]
        training_img = files["arr_8"]
        training_txt = files["arr_9"]
        max_label_len = max(valid_label_length.max() , train_label_length.max())



    # pad each output label to maximum text length
    
    train_padded_txt = pad_sequences(training_txt, maxlen=max_label_len, padding='post', value = len(char_list))
    valid_padded_txt = pad_sequences(valid_txt, maxlen=max_label_len, padding='post', value = len(char_list))

    dataset = {'valid_orig_txt' : valid_orig_txt,
    'valid_label_length' : valid_label_length,
    'valid_input_length' : valid_input_length,
    'valid_img' : valid_img,
    'valid_txt' : valid_txt,
    'orig_txt' : orig_txt,
    'train_label_length' : train_label_length,
    'train_input_length' : train_input_length,
    'training_img' : training_img,
    'training_txt' : training_txt,
    'max_label_len' : max_label_len,
    'train_padded_txt' : train_padded_txt,
    'valid_padded_txt' : valid_padded_txt}

    return dataset

# Model without weights
def create_base_model():
    # input with shape of height=32 and width=128 
    char_list = list('-0123456789')
    inputs = Input(shape=(32,128,1))
    
    # convolution layer with kernel size (3,3)
    conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
    pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
    batch_norm_1 = BatchNormalization()(pool_1)
    
    conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(batch_norm_1)
    pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
    batch_norm_2 = BatchNormalization()(pool_2)

    
    conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(batch_norm_2)
    conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)
    # poolig layer with kernel size (2,1)
    pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
    batch_norm_3 = BatchNormalization()(pool_4)


    
    conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_3)
    # Batch normalization layer
    batch_norm_5 = BatchNormalization()(conv_5)
    
    conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
    batch_norm_6 = BatchNormalization()(conv_6)
    pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
    
    conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)
    
    squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
    
    # bidirectional LSTM layers with units=128
    blstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(squeezed)
    blstm_2 = Bidirectional(LSTM(128, return_sequences=True, dropout = 0.2))(blstm_1)
    
    outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)

    # model to be used at test time
    return Model(inputs, outputs)

# Testing model
def test_model(act_model, data):
    from google.colab.patches import cv2_imshow
    import numpy as np
    char_list = list('-0123456789')
    
    valid_orig_txt = data["valid_orig_txt"]
    valid_label_length = data["valid_label_length"]
    valid_input_length = data["valid_input_length"]
    valid_img = data["valid_img"]
    valid_txt = data["valid_txt"]
    orig_txt = data["orig_txt"]
    train_label_length = data["train_label_length"]
    train_input_length = data["train_input_length"]
    training_img = data["training_img"]
    training_txt = data["training_txt"]
    max_label_len = data["max_label_len"]
    train_padded_txt = data["train_padded_txt"]
    valid_padded_txt = data["valid_padded_txt"]


    def unsig(a):
        if a[0] == "-":
            return a[1:]
        else:
            return a

    def numerical(a, b):
        return unsig(a) == unsig(b)




    # predict outputs on validation images
    prediction = act_model.predict(valid_img)
    # use CTC decoder
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                            greedy=True)[0][0])
    
    # see the results
    i = 0
    true_num = 0
    false_num = 0
    for x in out:
        asd = ""
        # print("original_text =  ", valid_orig_txt[i])
        # print("predicted text = ", end = '')
        for p in x:  
            if int(p) != -1:
                asd = asd + char_list[int(p)]
                # print(char_list[int(p)], end = '')       
        if numerical(asd, valid_orig_txt[i]):
            true_num = true_num+1
            # cv2_imshow((valid_img[i]*255).astype(int))
            # print("pred:", asd ,"true:", valid_orig_txt[i])
        else:
            false_num = false_num+1
            cv2_imshow((valid_img[i]*255).astype(int))
            print("pred:", asd ,"true:", valid_orig_txt[i], "number: ", i)
        # print('\n')
        i+=1

    print("Accuracy:", true_num/(true_num + false_num))

# Extracting image from the file
def extract_images(name, folder, test_max = 10000):
    import fitz
    doc = fitz.open(name)
    os.mkdir(folder)
    for i in range(min( len(doc), test_max)):
        for img in doc.getPageImageList(i):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            print(pix)
            if pix.n < 5:       # this is GRAY or RGB
                pix.writePNG(folder + "/p%s-%s.png" % (i, xref))
            else:               # CMYK: convert to RGB first
                pix1 = fitz.Pixmap(fitz.csRGB, pix)
                pix1.writePNG(folder + "/p%s-%s.png" % (i, xref))
                pix1 = None
            pix = None

# Take page from folder
def get_page(page_name):
    img = cv2.imread(page_name, cv2.IMREAD_GRAYSCALE)    
    return img

# Delete unused images
def del_images():
    try:
        shutil.rmtree('pages')
    except:
        pass

from random import randint

# Covert one sample to readable format
def to_num_one(sm):
    if len(sm) <= 1:
        return -1
    else:
        return float(sm[:-1] + "." + sm[-1])

assert to_num_one("00") == 0
assert to_num_one("124") == 12.4

# Covert all samples sample to readable format
def to_num(arr):
    import copy
    cd = copy.deepcopy(arr)
    i = 0
    
    for row in arr:
        j = 0
        for x in row:
            cd[i][j] = to_num_one(x)
            j = j + 1
        i = i + 1
    return cd

assert to_num([["01", "02"], ["03", "04"], ["03", "04"] ]) == [[0.1, 0.2], [0.3, 0.4], [0.3, 0.4]]

# Highlighting vertical lines
def only_vertical_lines(img_bin):
    import numpy as np
    kernel_len = np.array(img_bin).shape[1]//100
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))

    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
    cv2.imwrite("/Users/YOURPATH/vertical.jpg",vertical_lines)
    return image_1

# Highlighting horizontal lines
def only_horiztal_lines(img_bin):
    import numpy as np
    kernel_len = np.array(img_bin).shape[1]//100
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))

    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
    cv2.imwrite("/Users/YOURPATH/horizontal.jpg",horizontal_lines)
    return image_2


# Extracting all cells from page
def get_cells_per_page(page):
    from scipy.signal import find_peaks
    import numpy as np
    img = page.copy()
    img_bin = 255-img

    image_1 = only_vertical_lines(img_bin)
    ar = [image_1[:,i].sum() for i in range(img.shape[1])]
    x = np.array(ar)
    vertical_lines, _ = find_peaks(x, prominence = 10000, distance = 10)

    image_2 = only_horiztal_lines(img_bin)
    ar = [image_2[i,:].sum() for i in range(img.shape[0])]
    x = np.array(ar)
    horizontal_lines, _ = find_peaks(x, prominence = 7000, distance = 10)

    numbers = []
    distr = []

    arr = []

    for i in range(len(vertical_lines) - 1):
        arr.append([])
        for j in range(len(horizontal_lines) - 1):
            x1 = vertical_lines[i]
            x2 = vertical_lines[i+1]
            y1 = horizontal_lines[j]
            y2 = horizontal_lines[j+1]

        
            summ = np.sum(img_bin[y1:y2,x1:x2])
            cell = img[y1:y2,x1:x2].copy()

            arr[i].append(cell)
    return arr




# Prediction for one cell
def predict(cells):
    import numpy as np
    import copy
    char_list = list('-0123456789')
    ans = copy.deepcopy(cells)
    for i in range(len(cells)):
        for j in range(len(cells[i])):
            ans[i][j] = pred(cells[i][j])
    return ans

# Prediction for whole page
def pred( cell):
    import numpy as np
    imgg = cell.copy()
    char_list = list('-0123456789')
    imgg = cv2.resize(imgg, dsize=(128, 32), interpolation=cv2.INTER_CUBIC)

    imgg = imgg/255.
    prediction = act_model.predict( imgg[np.newaxis,:,:,np.newaxis] )

    
    
    out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                    greedy=True)[0][0])
    for x in out:
        asd = ""
        for p in x:  
            if int(p) != -1:
                asd = asd + char_list[int(p)]
    



    return to_num_one(asd)
    




dic_sign = {0:"-", 1:"+", 2:"0"}





