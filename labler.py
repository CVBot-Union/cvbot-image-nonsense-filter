import cv2
import os
from os import listdir
from os.path import isfile, join

from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
import argparse

classes = ['nonsense','people']

def predict(image_path):
    # This line must be executed before loading Keras model.
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    y_classes = preds.argmax(axis=-1)
    print(classes[int(y_classes)])
    y_dict = {}
    y_dict['human_class'] = classes[int(y_classes)]
    # y_dict['conf'] = preds[int(y_classes)]
    return y_dict


datapath = "./uncategorized/"

label_dir = {
    'p': './train/people/',
    'n': './train/nonsense/',
}

dataset_image = listdir(datapath)
model = load_model('./model.h5')

for image_name in dataset_image:
    image = cv2.imread(join(datapath,image_name))

    predict_dict = predict(join(datapath,image_name))

    # image = cv2.putText(image, 'Conf Score:' + predict_dict['conf'],(50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA)
    image = cv2.resize(image,(600,600))
    image = cv2.putText(image, 'Predict:' + predict_dict['human_class'],(50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0),2,cv2.LINE_AA)
    cv2.imshow('labler', image) 
    key = cv2.waitKey(0)
    print(key)
    try:
        if key == ord('p'):
            os.rename(join(datapath, image_name), join(label_dir['p'] + image_name))
        elif key == ord('n'):
            os.rename(join(datapath + image_name), join(label_dir['n'] + image_name))
        elif key == ord('x'):
            pass
        elif key == 113:
            break
    except FileExistsError as e:
        print(e)

cv2.destroyAllWindows()  
