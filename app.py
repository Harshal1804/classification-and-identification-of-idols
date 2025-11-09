# app.py — Divine Idol Classifier (Final + HuggingFace Fix)
# ---------------------------------------------------------
import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image
import json
from huggingface_hub import hf_hub_download

# ---- Safety Fixes ----
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"

# ---- Model + Class Info ----
MODEL_REPO = "Harshal1804/classification-and-identification-of-idols"  # ✅ must be your MODEL repo name (not Space!)
MODEL_FILE = "idols_classifier_fixed.keras"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    with st.spinner("🔽 Downloading model from Hugging Face Hub..."):
        model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
        model = tf.keras.models.load_model(model_path)
        st.success("✅ Model loaded successfully!")
        return model

@st.cache_resource
def load_class_names():
    with open(CLASS_NAMES_PATH) as f:
        return json.load(f)

model = load_model()
class_names = load_class_names()

st.set_page_config(page_title="Divine Idol Classifier", page_icon="🪔", layout="wide")

# ---- Styling ----
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Marcellus&display=swap');

html, body, [class*="stApp"] {
  font-family: 'Marcellus', serif;
  color:#2b1a09;
  background: linear-gradient(270deg,#9c6644,#b97a57,#e8c07d,#9c6644);
  background-size:800% 800%;
  animation: gradientShift 25s ease infinite;
}
@keyframes gradientShift {
  0%{background-position:0% 50%;}
  50%{background-position:100% 50%;}
  100%{background-position:0% 50%;}
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #8b5e3c, #b87b4b, #e8c07d);
  border-right: 3px solid rgba(255,215,128,0.5);
  box-shadow: 2px 0 15px rgba(0,0,0,0.3);
  color: #fffbe7;
}
section[data-testid="stSidebar"] h1, 
section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3 {
  color: #ffe9a0 !important;
  text-shadow: 0 0 10px rgba(255,230,150,0.8);
}
section[data-testid="stSidebar"] label, 
section[data-testid="stSidebar"] p, 
section[data-testid="stSidebar"] span {
  color: #fff3b0 !important;
}
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("""
<header style="
  text-align:center;
  padding:1.5rem 0;
  border-radius:15px;
  background:linear-gradient(90deg,#8b4513,#b87333,#d4a76a,#b87333,#8b4513);
  box-shadow:0 5px 20px rgba(0,0,0,0.4);
  border:3px solid rgba(255,215,128,0.5);
">
  <h1 style="margin:0;color:#fff8dc;text-shadow:0 0 15px rgba(255,210,100,0.9);
             font-size:2.6rem;letter-spacing:1px;">
    🕉️ Divine Idol Classifier
  </h1>
  <p style="margin-top:0.4rem;color:#ffeaa7;font-size:1.2rem;
            text-shadow:0 0 10px rgba(255,240,180,0.9);">
    Enlightened by AI — Discover Vishnu, Mahadev, or Sursundari 🔱
  </p>
</header>
""", unsafe_allow_html=True)

# ---- Sidebar ----
st.sidebar.header("⚙️ App Controls")
mode = st.sidebar.radio("Choose mode:", ["Single Image", "Compare Two Images"])
st.sidebar.info("Built with 🧠 TensorFlow + Streamlit\n\nby **Harshal Shinde**")
st.sidebar.caption("Dynamic Temple Theme • 2025")

# ---- Prediction Function ----
def predict(img_data):
    img = img_data.resize(IMG_SIZE)
    arr = image.img_to_array(img)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr)
    top_idx = preds[0].argsort()[-3:][::-1]
    return preds[0], [(class_names[i], preds[0][i]) for i in top_idx]

# ---- Idol Info ----
idol_info = {
    "vishnu": {"text": "Preserver of life, symbol of peace and cosmic balance.",
               "bg": "linear-gradient(135deg, #b3e5fc, #81d4fa, #4fc3f7)"},
    "mahadev": {"text": "Destroyer of evil, the supreme ascetic and source of renewal.",
                "bg": "linear-gradient(135deg, #d7ccc8, #b0bec5, #90a4ae)"},
    "sursundari": {"text": "Symbol of divine grace, art, and beauty.",
                   "bg": "linear-gradient(135deg, #f8bbd0, #f48fb1, #f06292)"}
}

# ---- Modes ----
if mode == "Single Image":
    file = st.file_uploader("📸 Upload Idol Image", type=["jpg","jpeg","png"])
    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded Idol", use_container_width=True)
        st.subheader("🔮 Prediction Results")
        probs, top3 = predict(img)
        main_label, main_conf = top3[0]
        key = main_label.lower().strip()
        info = idol_info.get(key, {"text": "No information available.",
                                   "bg": "linear-gradient(145deg,#f8e7c1,#e7c991)"})
        st.markdown(f"<p style='color:#ffef9f;font-size:1.4rem;text-shadow:0 0 10px gold;'>✨ Most likely: <b>{main_label}</b> ({main_conf*100:.2f}%)</p>", unsafe_allow_html=True)
        st.markdown(f"<div style='margin-top:18px;border-radius:12px;padding:1rem;text-align:center;background:{info['bg']};color:#2b1a09;border:2px solid rgba(255,215,128,0.7);box-shadow:0 4px 15px rgba(0,0,0,0.25);'><b>{main_label}</b>: {info['text']}</div>", unsafe_allow_html=True)

elif mode == "Compare Two Images":
    c1, c2 = st.columns(2)
    with c1:
        f1 = st.file_uploader("📷 Upload Image 1", type=["jpg","jpeg","png"], key="1")
    with c2:
        f2 = st.file_uploader("📸 Upload Image 2", type=["jpg","jpeg","png"], key="2")
    if f1 and f2:
        img1 = Image.open(f1).convert("RGB")
        img2 = Image.open(f2).convert("RGB")
        st.image([img1, img2], caption=["Image 1", "Image 2"], use_container_width=True)
        p1, t1 = predict(img1)
        p2, t2 = predict(img2)
        st.markdown(f"<p style='color:#ffef9f;font-size:1.2rem;'>✨ Image 1: {t1[0][0]} ({t1[0][1]*100:.2f}%)</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#ffef9f;font-size:1.2rem;'>✨ Image 2: {t2[0][0]} ({t2[0][1]*100:.2f}%)</p>", unsafe_allow_html=True)

# ---- Footer ----
st.markdown("""
<footer style="
  width:100%;
  text-align:center;
  color:#fff8dc;
  background:linear-gradient(90deg,#8b4513,#b87333,#d4a76a,#b87333,#8b4513);
  padding:1rem 0;
  border-radius:15px;
  margin-top:3rem;
  box-shadow:0 -4px 15px rgba(0,0,0,0.4);
  text-shadow:0 0 8px rgba(255,220,120,0.7);
  font-size:0.95rem;
">
  🪔 <b>Idol Classifier</b> • Crafted with devotion by <b>Harshal Shinde</b><br>
  © 2025 • Temple-Themed AI • Powered by TensorFlow & Streamlit
</footer>
""", unsafe_allow_html=True)

