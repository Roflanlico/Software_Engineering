import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from io import BytesIO
from PIL import Image
import io
import streamlit as st 
from tensorflow.keras.utils import img_to_array

def load_image():
    uploaded_file = st.file_uploader( label='Выберите изображение для распознавания')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

@st.cache(allow_output_mutation=True)
def load_model():
    model = keras.applications.VGG16()
    return model

def preprocess_image(img):
    img = img.resize((224, 224))
    #x = img_to_array(img)
    x= np.asarray(img)
    x = keras.applications.vgg16.preprocess_input(img)
    x = np.expand_dims(x, axis=0)
    return x

st.title('Классификация изображений')
model = load_model()
img = load_image()

result = st.button('Распознать изображение')
if result:
    x = preprocess_image(img)
    res = model.predict(x)
    st.write('**Номер результирующего класса:**')
    print(np.argmax(res))
