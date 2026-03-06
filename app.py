import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

IMG_SIZE = 224

st.title("Deepfake Detection System")

model = load_model("deepfake_detector_model.h5")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image")

    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img = img/255.0
    img = np.reshape(img,(1,IMG_SIZE,IMG_SIZE,3))

    prediction = model.predict(img)

    if prediction[0][0] > 0.5:
        st.error("FAKE IMAGE DETECTED")
    else:
        st.success("REAL IMAGE")