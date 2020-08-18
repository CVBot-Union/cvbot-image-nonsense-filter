from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import argparse

classes = ['nonsense','people']

def predict(fpath,image_path):
    model = load_model(fpath)
    # This line must be executed before loading Keras model.
    img = image.load_img(image_path, target_size=(139, 139))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    y_classes = preds.argmax(axis=-1)
    print(classes[int(y_classes)])
    y_dict = {}
    y_dict['human_class'] = classes[int(y_classes)]
    y_dict['conf'] = preds[int(y_classes)]
    return y_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m","--model",type=str, help="model")
    parser.add_argument("-i","--image",type=str, help="image")
    args = parser.parse_args()
    predict(args.model,args.image)