import argparse
import json

import numpy as np
import requests
from tensorflow.keras.preprocessing import image

# Argument parser for giving input image_path from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path of the image")
args = vars(ap.parse_args())

image_path = args['image']
# Preprocessing our input image
img = image.img_to_array(image.load_img(image_path, target_size=(224, 224))) / 255.

payload = {
    "instances": [{'resnet152v2_input': img.tolist()}]
}

# sending post request to TensorFlow Serving server
r = requests.post('http://localhost:9000/v1/models/Model:predict', json=payload)
pred = json.loads(r.content.decode('utf-8'))

# Decoding the response
# decode_predictions(preds, top=5) by default gives top 5 results
# You can pass "top=10" to get top 10 predicitons
print(pred)