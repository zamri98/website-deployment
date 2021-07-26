# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 13:40:46 2021

@author: risingkillersz42
"""

import streamlit as st
import tensorflow as tf


st.set_option("deprecation.showfileUploaderEncoding", False)
@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('my_model2.hdf5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Covid X-ray Classification
         """
         )

file = st.file_uploader("Please upload only an x-ray scan file", type=["jpg", "png"])
#import cv2
from PIL import Image, ImageOps
import numpy as np
#st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        size = (180,180)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        img = np.asarray(image)
        
        #img_reshape=img[np.newaxis,...]
        #img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
    
        prediction = model.predict(img_reshape)
        
        return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    class_names = ['COVID','Normal']
    string = "This X-ray image most likely is {} with a {:.2f} percent confidence." .format(class_names[np.argmax(predictions)], 100*np.max(score))
    st.success(string)
    
