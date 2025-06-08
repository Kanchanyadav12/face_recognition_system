import streamlit as st
from keras_facenet import FaceNet
import numpy as np
from PIL import Image
from numpy.linalg import norm
import pickle

st.title("🎯 Face Recognition App")
st.markdown("Upload a face image and get the predicted identity.")

# Load model and known encodings
embedder = FaceNet()
with open("encodings.pkl", "rb") as f:
    known_faces = pickle.load(f)

# File uploader
uploaded_file = st.file_uploader("Upload a Face Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate embedding
    embedding = embedder.embeddings([np.array(image)])[0]

    # Match with known faces
    min_dist = 100
    identity = "Unknown"
    for name, db_enc in known_faces.items():
        dist = norm(embedding - db_enc)
        if dist < min_dist:
            min_dist = dist
            identity = name if dist < 10 else "Unknown"

    st.success(f"Predicted Identity: {identity} (Distance: {min_dist:.2f})")
