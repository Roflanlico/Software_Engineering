import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from io import BytesIO
from PIL import Image
import io
from fastapi import FastAPI
from pydantic import BaseModel


def load_image():
    img = Image.open('test_image.jpg')
    return img

def load_model():
    model = keras.applications.VGG16()
    return model

def preprocess_image(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = keras.applications.vgg16.preprocess_input(img)
    x = np.expand_dims(x, axis=0)
    return x

class Item(BaseModel):
    text: str
    
app = FastAPI()

model = load_model()
img = load_image()

x = preprocess_image(img)
res = model.predict(x)

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/predict/")
def predict(item: Item):
    return {"Номер результирующего класса": np.argmax(res)}















