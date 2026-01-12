import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

def predict_image(model_path, image_path):
    model = load_model(model_path)
    img = Image.open(image_path).resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    result = "Agricultural" if prediction[0][0] > 0.5 else "Non-Agricultural"
    return result, prediction[0][0]