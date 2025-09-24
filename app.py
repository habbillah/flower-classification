import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os

 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

@st.cache_resource
def load_flower_model():
    model_path = "flower_classification_model_sequential.h5"
    try:
        
        model = tf.keras.models.load_model(model_path, compile=False)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        return None


 
 
model = load_flower_model()

 
class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

st.title("🌸 Flower Classification App")
st.write("Upload gambar bunga, dan model MobileNetV2 akan memprediksi jenis bunganya.")

 
uploaded_file = st.file_uploader("Pilih gambar bunga...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Gambar diupload", use_column_width=True)

       
        img = img.resize((224, 224))   
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

       
        with st.spinner("Memproses..."):
            predictions = model.predict(img_array, verbose=0)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = np.max(predictions[0]) * 100

        st.success(f"Prediksi: **{predicted_class}** ({confidence:.2f}% yakin)")
        
        
        st.write("### Detail Probabilitas:")
        for i, (class_name, prob) in enumerate(zip(class_names, predictions[0])):
            percentage = prob * 100
            st.write(f"**{class_name}**: {percentage:.2f}%")
            
    except Exception as e:
        st.error(f"Error saat pemrosesan: {str(e)}")
        
elif model is None:
    st.error("❌ Model tidak dapat dimuat. Silakan periksa file model Anda.")