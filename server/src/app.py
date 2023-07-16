import os
import json
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
from torchvision import transforms, datasets
from flask_cors import CORS, cross_origin
from classes import *

app = Flask(__name__)
CORS(app)

# Define the model path
model_path = os.path.join(".", "server/src/resnet50_trained.pth")
# model_path = os.path.join(".", "resnet50_trained.pth")

# Load the trained model
model = torch.load(model_path, map_location='cpu')
model.eval()  # Set the model to evaluation mode

@app.route('/', methods=['GET', 'POST'])
# @cross_origin()
def home():
    context = { }
    
    if request.method == 'POST':
        file = request.files.get('image')
        if file is None:
            return jsonify({'error': 'no file'}), 400
        
        # Define the transformation
        transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        image = Image.open(file.stream)
        image = transform(image)
        image = image.unsqueeze(0)

        with torch.no_grad():
            output = model(image)

        _, predicted_idx = torch.max(output, 1)
        predicted_class = idx_to_class[predicted_idx.item()]
        class_idx, class_name = predicted_class.split("-")
        class_name = class_name.replace("_", " ")

        return jsonify({'predicted_class': class_name})

    else:
        return render_template('index.html', context=context)

@app.route('/api')
def api():
    return jsonify('Testing api call')

# @app.route('/upload', methods=['POST', 'GET'])
# def upload():
#     try:
#         imageFile = Flask.request.files.get('imageFile', '')
#     except Exception as err:


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))