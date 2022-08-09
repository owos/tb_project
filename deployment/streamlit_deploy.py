import numpy as np
import cv2 as cv 
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import streamlit as st
from styling import footer

st.cache(allow_output_mutation=True)
st.title('TB Image Classifier') 
#
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
# loading model
model_path = 'tb_model'
model = load_model(model_path)
#loading the imaage

file = st.file_uploader('Upload the image', type=['png', 'jpg'], accept_multiple_files=False, key=None, help=None, 
                                on_change=None, args=None, kwargs=None)

run = st.button('Make Prediction', key=None, help=None, on_click=None, args=None, kwargs=None)
st.subheader('This app classifies a x-ray iamge if it has TB or not')
#image laoder
def load_image (img_path, img_size, show=False):
    img = image.load_img(img_path, target_size=img_size)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)      #expanding image tensor
    img_tensor = img_tensor /255.           # scaling the image_T

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()
    return img_tensor


img_size = (300, 300)
img_path = 'inference image from medscape.jpg'

classes = ['Normal', 'Tuberculosis']
if __name__ == "__main__":
    ## load img
    footer()
    if run == True:
        if file is not None:
            img = load_image(img_path, img_size)
            pred = model.predict(img)
            output = classes[round(pred[0][0])] 
            st.subheader(f'The image is {output}')
        else: 
            st.write("Please upload an image first")
    

            #st.image(file)
            
            


