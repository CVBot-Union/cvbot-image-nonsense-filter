import json
import requests
import os
import numpy as np
import shutil
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from expiringdict import ExpiringDict
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

UPLOAD_FOLDER = 'upload/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024
classes = ['nonsense','people']
BUCKET_URL = "https://twipush-s3.s3-ap-northeast-1.amazonaws.com/images/"
cache = ExpiringDict(max_len=256, max_age_seconds=6 * 60 * 60)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Uncomment this line if you are making a Cross domain request
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict/twipush', methods=['GET'])
def twipush_classifier():
    if request.args.get('media_id') is None:
        return jsonify({'error':True,'msg':'No File Name'})
    is_cache_miss = False
    if cache.get(request.args.get('media_id')) is None:
        is_cache_miss = True
        print(request.args.get('media_id') + ': Cache Miss.')
        r = requests.get(BUCKET_URL + request.args.get('media_id') + '.png', stream=True)
        if r.status_code == 200:
            save_filename = UPLOAD_FOLDER + request.args.get('media_id') + '.png'
            with open(save_filename, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
        else:
            return jsonify({'error':True,'msg':'Error While Downloading File'})
    
    if is_cache_miss:
        img = image.img_to_array(image.load_img(save_filename, target_size=(224, 224))) / 255
        os.remove(save_filename)
        # Creating payload for TensorFlow serving request
        payload = {
            "instances": [{'resnet152v2_input': img.tolist()}]
        }

        # Making POST request
        r = requests.post('http://localhost:9000/v1/models/Model:predict', json=payload)
        # Decoding results from TensorFlow Serving server
        pred = json.loads(r.content.decode('utf-8'))
        infer_result = np.array(pred['predictions'])
        infer_dict = {
            'complete_inf': infer_result.tolist(),
            'best_inf': infer_result.argmax(axis=-1).tolist()
        }
        cache[request.args.get('media_id')] = infer_dict
    else:
        print(request.args.get('media_id') + ': Cache Hit.')
        infer_dict = cache.get(request.args.get('media_id'))
    
    resp = jsonify({'error':False,'msg':infer_dict})
    resp.headers['X-InferCache-Miss'] = is_cache_miss
    return resp




@app.route('/predict', methods=['POST'])
def image_classifier():
    if 'file' not in request.files:
        return jsonify({'error':True,'msg':'No File Field'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error':True,'msg':'No File Name'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_filename)
        # Decoding and pre-processing base64 image
        img = image.img_to_array(image.load_img(save_filename, target_size=(139, 139))) / 255
        os.remove(save_filename)

        # Creating payload for TensorFlow serving request
        payload = {
            "instances": [{'inception_v3_input': img.tolist()}]
        }

        # Making POST request
        r = requests.post('http://localhost:9000/v1/models/Model:predict', json=payload)

        # Decoding results from TensorFlow Serving server
        pred = json.loads(r.content.decode('utf-8'))
        infer_result = np.array(pred['predictions'])
        return jsonify({'error':False,'msg': {
            'complete_inf': infer_result.tolist(),
            'best_inf': infer_result.argmax(axis=-1).tolist()
            }
        })

if __name__ == '__main__':
    WSGIServer(('0.0.0.0', 5000), app).serve_forever()
