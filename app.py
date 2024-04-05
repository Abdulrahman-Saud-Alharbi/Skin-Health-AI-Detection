from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

app = Flask(__name__)

# Load the model
model = load_model('skin_disease_model.h5')

file_path = r'extracted_directories.txt'  # Replace with the actual path to 'extracted_directories.txt'

extracted_directories = []

# Open the file and read the lines
with open(file_path, 'r') as file:
    extracted_directories = [line.strip() for line in file]

# Manually define the class label mapping based on your training
class_label_map = {i: name for i, name in enumerate(extracted_directories)}

def predict_skin_disease(image_path, model, class_label_map, img_height=150, img_width=150):
    # Load and resize the image
    img = load_img(image_path, target_size=(img_height, img_width))

    # Convert the image to a numpy array
    img_array = img_to_array(img)

    # Scale the image (as done in training)
    img_array = img_array / 255.0

    # Expand dimensions to match the model's input format
    img_array = np.expand_dims(img_array, axis=0)

    # Make a prediction
    prediction = model.predict(img_array)

    # Retrieve the class index
    class_index = np.argmax(prediction, axis=1)

    # Map class index to class label
    class_label = class_label_map[class_index[0]]

    return class_label

@app.route('/', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None  # Change 'result' to 'prediction'
    if request.method == 'POST':
        # Get the file from the POST request
        file = request.files['file']
        if file:
            # Save the file to a temporary location
            file_path = '/tmp/temp_image.jpg'
            file.save(file_path)

            # Predict the skin disease
            prediction = predict_skin_disease(file_path, model, class_label_map)

    return render_template('index.html', prediction=prediction)  # Change 'result' to 'prediction'

if __name__ == '__main__':
    app.run(debug=True)
