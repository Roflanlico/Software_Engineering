import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from io import BytesIO
from PIL import Image
import io
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.utils import img_to_array

def load_image(imageName):
    if imageName == "volcano":
        return Image.open('test_image.jpg')
    else:
        return Image.open('test_image2.jpg')

def load_model():
    model = keras.applications.VGG16()
    return model

def preprocess_image(img):
    img = img.resize((224, 224))
    x = np.array(img)
    x = keras.applications.vgg16.preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    return x

def predictImg(imageName):
    model = load_model()
    img = load_image(imageName)
    x = preprocess_image(img)
    res = model.predict(x)
    return int(np.argmax(res))

class Item(BaseModel):
    text: str
    
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/predict/")
def predict(item: Item):
    return predictImg(item.text)















