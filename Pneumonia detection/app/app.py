from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os

app = Flask(__name__)

# Load your trained model
model = load_model('C:/Users/91981/Desktop/major-1/saved_model/final.h5')

def apply_clahe_to_folder(input_folder, output_folder, clip_limit=2.0, grid_size=(8, 8)):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through subfolders (classes)
    for class_folder in os.listdir(input_folder):
        class_folder_path = os.path.join(input_folder, class_folder)
        output_class_folder_path = os.path.join(output_folder, class_folder)

        # Create output class folder if it doesn't exist
        if not os.path.exists(output_class_folder_path):
            os.makedirs(output_class_folder_path)

        # Iterate through images in the class folder
        for filename in os.listdir(class_folder_path):
            input_image_path = os.path.join(class_folder_path, filename)
            output_image_path = os.path.join(output_class_folder_path, filename)

            # Read the image
            image = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)

            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
            clahe_image = clahe.apply(image)

            # Save the processed image
            cv2.imwrite(output_image_path, clahe_image)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

@app.route('/')
def index():
    return render_template('code.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', result="No file part")
    
    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', result="No selected file")
    
    if file:
        img_path = f"C:/Users/91981/Desktop/major-1/app/static/{file.filename}"
        file.save(img_path)
        apply_clahe_to_folder("static/","static/",2.0,(8,8))
        # Preprocess the uploaded image
        img_array = preprocess_image(img_path)
        
        # Make prediction
        prediction = model.predict(img_array)
        result = "Pneumonia" if prediction[0][0] > 0.5 else "Normal"

        return render_template('code.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
