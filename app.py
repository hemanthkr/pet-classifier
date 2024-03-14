from fastai.vision.all import *
import gradio as gr

def is_cat(x):
    return x[0].isupper()

learn = load_learner('model.pkl')

categories = ('Dog', 'Cat')

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))


iface = gr.Interface(fn=classify_image, inputs="image", outputs="label")
iface.launch()