import streamlit as st

from keras.models import load_model
import tensorflow_hub as hub
import tensorflow as tf

import io
import os
import cv2
import numpy as np

from PIL import Image



#MODEL_PATH = "model//Alzheimer_Cassifier.pkl"
MODEL_PATH = 'model/MyRiceModel_h5'
#LABELS_PATH = "model//model_classes.txt"
LABELS_PATH = 'model/model_classes.txt'


#ส่วนอัพโหลดรูป
def load_image():
    uploaded_file = st.file_uploader(label='Pick an MRI image to predict', type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        image = Image.open(uploaded_file)
        img   = np.array(image)
        #img = cv2.imread(img)
        img = cv2.resize(img,(224,224))
        img = np.reshape(img,[1,224,224,3])
        image = img
        return image
        #return Image.open(io.BytesIO(image_data))
    else:
        return None

        
        
#ส่วนเรียกใช้โมเดล
def load_model(model_path):
    #model ="ff"
    #model = pickle.load(open(model_path, 'rb'))
    #model = joblib.load(model_path)
    #model = joblib.load(open(os.path.join(MODEL_PATH),"rb"))
    #return model
    #model = tf.keras.models.load_model(model_path)
    #model = tf.keras.models.load_model(MODEL_PATH)
    model = tf.keras.models.load_model(
       MODEL_PATH,
       custom_objects={'KerasLayer':hub.KerasLayer}
)
    #model = torch.load(model_path, map_location='cpu')
    #model.eval()
    #model.fit(0.1,0.8)
    #result = model.scroe()
    #loss, acc_h5 = loaded_model_h5.evaluate(x_test, y_test, verbose=1)
    return model


#ส่วนเรียกไฟล์ที่เก็บคลาส
def load_labels(labels_file):
    with open(labels_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        return categories


def predict(model, categories, image):
    #model.compile(optimizer="adam",loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['acc'])
    #img = cv2.imread("5.jpg")
    #img = cv2.resize(img,(224,224))
    #img = np.reshape(img,[1,2)24,224,3]

    pred = model.predict(image)
    predict_output = np.argmax(pred, axis=1)
    if predict_output[0] == 0 :
        st.write('This is Arborio')
    if predict_output[0] == 1 :
        st.write('This is Basmati')
    if predict_output[0] == 2 :
        st.write('This is Ipsala')
    if predict_output[0] == 3 :
        st.write('This is Jasmine')
    if predict_output[0] == 4 :
        st.write('This is Karacadag')
    return 1

def main():
    st.title("Rice Classification 🌾")
    st.write('A Part Of Neural Network 1/2022')
    model = load_model(MODEL_PATH)
    categories = load_labels(LABELS_PATH)
    image = load_image()
    result = st.button('Run on image')
    if result:
        st.write('Calculating results...')
        #st.write(categories)
        predict(model, categories, image)


if __name__ == '__main__':
    main()
