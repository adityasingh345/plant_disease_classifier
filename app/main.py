import os 
import json 
from PIL import Image

import numpy as np 
import tensorflow as tf 
import streamlit as st

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/plant_disease_prediction_model_.h5"
# load the pre-trained model 
model = tf.keras.models.load_model(model_path)

 # loading the class names 
class_indices = json.load(open(f"{working_dir}/class_indices.json"))
 
 # function to load and pre-process the Image using pillow 
 
def load_preprocess_image(image_path, target_size=(224, 224)):
    # loading the image 
    img = Image.open(image_path)
    #resize the image
    img = img.resize(target_size)
    # convert the array into numpy array 
    img_array = np.array(img)
    #add batch dimension 
    img_array = np.expand_dims(img_array, axis= 0)
    # scaling the image between  0 to 1
    img_array = img_array.astype('float32') / 255. 
    return img_array 

def predict(model, image_path, class_indices):
    preprocessed_img = load_preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    predicted_class_index= np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

 # streamlit app  
st.title('Plant Disease classifier')

uploaded_img = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    image = Image.open(uploaded_img)
    col1, col2 = st.columns(2)
    
    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)
        
    with col2:
        if st.button('classify'):
            # preprocess the uploaded image and predict the class
            prediction = predict(model, uploaded_img, class_indices)
            st.success(f'prediction : {str(prediction)}')
            
            
            