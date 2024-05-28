import os
from flask import Flask, request, jsonify, render_template, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

model_path = r"C:\Users\Chun Yu\Desktop\123\banana_classifier.h5"
model = load_model(model_path, compile=False)

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/classify', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    file_path = os.path.join(os.getcwd(), file.filename)
    file.save(file_path)

    img = prepare_image(file_path)
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction[0])

    classes = ['過熟', '成熟', '爛掉', '未成熟']
    result = classes[predicted_class]

    return render_template('index.html', result=result, img_path=file.filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(os.getcwd(), filename)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
