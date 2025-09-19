import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# === CONFIG ===
MODEL_PATH = "plant_model.h5"
TRAIN_DIR = r"D:\indibreed\train"   # update this path if needed
IMG_SIZE = (224, 224)

# === Page Setup ===
st.set_page_config(page_title="Cattle Breed Classifier", page_icon="🐄", layout="wide")
st.image("logo.jpg", width=120)   # adjust width


# === Custom CSS (Colorful UI, with Background Color) ===
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(120deg, #d4fc79 0%, #96e6a1 100%);
            font-family: 'Segoe UI', sans-serif;
        }
        h1 {
            text-align: center;
            color: #1b4332;
            font-size: 40px;
        }
        .prediction-card {
            background: rgba(255, 255, 255, 0.9);
            padding: 25px;
            border-radius: 20px;
            box-shadow: 0px 6px 18px rgba(0,0,0,0.2);
            margin-top: 20px;
        }
        .breed-detail {
            background: #ffeaa7;
            margin: 6px 0;
            padding: 8px 12px;
            border-radius: 10px;
            color: #2d3436;
            font-size: 16px;
        }
        .confidence-bar {
            height: 22px;
            border-radius: 10px;
            background: #dfe6e9;
            margin-top: 10px;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            background: linear-gradient(to right, #6c5ce7, #00cec9);
            text-align: center;
            color: white;
            font-weight: bold;
            font-size: 12px;
            line-height: 22px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# === Load class names ===
@st.cache_resource
def load_class_names(train_dir):
    return sorted([
        folder for folder in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, folder))
    ])

class_names = load_class_names(TRAIN_DIR)

# === Load model ===
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# === Breed Details Dictionary (UNCHANGED) ===
breed_details = {
    "GIR Male": {
        "Age": "15 - 12 YEARS ",
        "Weight": "385 kg",
        "Colours": "chocolate brown patches, pure red, black ",
        "SPL FEATURES:": "hors are. peculiarly curved giving half moon appearence ",
        "Milk Yield": "No milk",
        "Origin": "Gir forest in Maharastra, Rajasthan near Gujarat",
    },
    "GIR Female": {
        "Age": "15 - 12 YEARS ",
        "Weight": "385 kg",
        "Colours": "chocolate brown patches, pure red, black ",
        "SPL FEATURES:": "hors are. peculiarly curved giving half moon appearence ",
        "Milk Yield": "1200- 2000 KG/lactation",
        "Origin": "Gir forest in Maharastra, Rajasthan near Gujarat",
    }, 
    "Sahiwal Male": {
        "Age": "20 - 24 YEARS ",
        "Weight": "400-550 kg",
        "Colours": "Pale red/red, flashed with white patches",
        "SPL FEATURES:": "best indigeneous dairy breed, heavy breed, loose skin, symmetrical body",
        "Milk Yield": "No milk",
        "Origin": "Montagomery",
    },
    "Sahiwal Female": {
        "Age": "20 - 24 YEARS ",
        "Weight": "400-500 kg",
        "Colours": "Pale red/red, flashed with white patches",
        "SPL FEATURES:": "best indigeneous dairy breed, heavy breed, loose skin, symmetrical body",
        "Milk Yield": "1400-2500 kgs per lactation",
        "Origin": "Montagomery",
    },
    # ➡️ keep rest breeds as in your code...
    "PULLIKULAM": {
        "Age": "20 YEARS ",
        "Weight": "Approx 325 -385 kg",
        "Colours": "Presence of reddish or brownish spots in muzzle, eyes, switch and back is the characteristic feature of this breed. grey or dark grey in colour.",
        "SPL FEATURES:": "Well- developed hump.Mainly used for penning in the field.Useful for ploughing. ",
        "Milk Yield": "no milk",
        "Origin": "Madurai district in Tamil Nadu",
    },
}

# === Prediction Function (Top-3) ===
def predict(image: Image.Image, top_k=3):
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = image.resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)[0]

    # Get top-k predictions
    top_indices = predictions.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        breed_name = class_names[idx]
        confidence = float(predictions[idx])
        details = breed_details.get(breed_name, {"Info": "Details not available"})
        results.append((breed_name, confidence, details))

    return results


# === UI ===
st.title("🐄 Cattle Breed Classifier")
st.markdown("### Upload or capture a cattle image to identify its breed and view details.")

option = st.radio("Choose input method:", ("📁 Upload an image", "📷 Use camera"))

image = None
if option == "📁 Upload an image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
elif option == "📷 Use camera":
    img_data = st.camera_input("Take a photo of the cattle")
    if img_data:
        image = Image.open(img_data)


# === Prediction Output (Top-3) ===
if image:
    col1, col2 = st.columns([2, 3])

    with col1:
        st.image(image, caption="📸 Input Image", use_container_width=True)

    with col2:
        with st.spinner("🔎 Analyzing..."):
            results = predict(image, top_k=3)

        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.subheader("✅ Top 3 Predictions")

        for i, (breed, conf, details) in enumerate(results):
            if i == 0:
                st.write(f"🏆 **Most Likely Breed:** {breed}")
            else:
                st.write(f"**Alternative Breed:** {breed}")

            # Confidence bar
            st.markdown(
                f"""
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {conf*100:.2f}%;">
                        {conf*100:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader("📋 Breed Details")
            for k, v in details.items():
                st.markdown(f"<div class='breed-detail'><b>{k}</b> {v}</div>", unsafe_allow_html=True)

            st.markdown("---")  # separator between predictions

        st.markdown('</div>', unsafe_allow_html=True)
