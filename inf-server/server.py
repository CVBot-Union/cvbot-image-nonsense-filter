import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
from flask import Flask, jsonify, request, render_template, url_for
from werkzeug.utils import secure_filename
from io import BytesIO
import numpy as np

UPLOAD_FOLDER = 'upload/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

tf.keras.backend.set_learning_phase(0)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024
classes = ['nonsense','people']


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def netInfer(filepath):
    model = tf.keras.models.load_model('../model.h5',compile=False)
    img = image.load_img(filepath, target_size=(139, 139))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error':True,'msg':'No File Field'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error':True,'msg':'No File Name'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_filename)
        infer_result = netInfer(save_filename)
        return jsonify({'error':False,'msg': {
            'complete_inf': infer_result.tolist(),
            'best_inf': infer_result.argmax(axis=-1).tolist()
            }
        })

if __name__ == '__main__':
    app.run()