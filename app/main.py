from flask import Flask, render_template, request, jsonify
from app.torch_utils import get_prediction, transform_image

app = Flask(__name__)

allowed_extensions = {"jpg", "jpeg", "png"}


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if file is None or file.filename == "":
            return jsonify({'error': 'no file'})
        if not allowed_file(file.filename):
            return jsonify({'format error': 'format not supported'})
        try:
            img_bytes = file.read()
            image_tensor = transform_image(img_bytes)
            prediction = get_prediction(image_tensor)
            data = {'prediction': prediction.item(), 'class_name': str(prediction.item())}  # because we can not put tensor in json
            return jsonify(data)
        except:
            return jsonify({'error': 'error during prediction'})


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions