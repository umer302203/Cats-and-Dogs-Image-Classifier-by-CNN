import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('cats_dogs_model.keras')


def predict_image(img):
    
    img = img.resize((150, 150))

    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)
    
    
    prediction = model.predict(img_array)[0][0]
    
    
    if prediction > 0.5:
        return {"Dog": float(prediction), "Cat": float(1 - prediction)}
    else:
        return {"Cat": float(1 - prediction), "Dog": float(prediction)}


interface = gr.Interface(
    fn=predict_image,             
    inputs=gr.Image(type="pil"),   
    outputs=gr.Label(num_top_classes=2), 
    title="Cat vs Dog Classifier 🐱🐶",
    description="Upload an image of a Cat or a Dog and let the AI classify it!"
)


interface.launch()
