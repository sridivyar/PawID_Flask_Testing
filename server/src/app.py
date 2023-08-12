import os
import json
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
from torchvision import transforms, datasets
from flask_cors import CORS, cross_origin
from classes import * 
import tensorflow as tf
# from tensorflow.keras.applications import MobileNetV2
import numpy as np

app = Flask(__name__)
CORS(app)

# Define the model paths
# model_path = os.path.join(".", "resnet50_trained.pth")
model_path = os.path.join(".", "efficientnet_b0_trained.pth") #Dog breed classifier
model1_path = os.path.join(".", "Mobilenet.h5")               #Dog/non-dog classifier

# Load the trained models
model1 = tf.keras.models.load_model(model1_path)
model = torch.load(model_path, map_location='cpu')
model.eval()  # Set the model to evaluation mode

#Define function to first determine if image is a dog or not
def predict_dog(image_path):
    # img = tf.keras.preprocessing.image.load_img(image_path, target_size=(32, 32))
    img = tf.keras.preprocessing.image.smart_resize(image_path, (32,32))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, 0)

    prediction = model1.predict(img_array)
    print(prediction)
    return "Dog" if prediction[0][0] > 0.12 else "Not Dog"

@app.route('/', methods=['GET', 'POST'])
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
        image = image.convert("RGB")

        
        #First check if image is dog or not
        result = predict_dog(image)

        
        if result != "Dog":
            # raise ValueError("The prediction is not 'Dog'. Aborting further tests.")
            return jsonify({'predicted_class': "Please upload a photo of a dog"})
        else:
            
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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=int(os.environ.get('PORT', 8080)))