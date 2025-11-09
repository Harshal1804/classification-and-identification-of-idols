# app.py — Divine Idol Classifier (Final Ancient Themed + Styled Sidebar)
# ----------------------------------------------------------------------
import os
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image
import json

# ---- Safety Fixes ----
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
import tensorflow as tf
import requests
import os
import streamlit as st

MODEL_URL = "https://huggingface.co/spaces/Harshal1804/classification-and-identification-of-idols/blob/main/idols_classifier_fixed.keras"
MODEL_PATH = "idols_classifier_fixed.keras"

@st.cache_resource
def load_model():
    # Download model if not already present locally
    if not os.path.exists(MODEL_PATH):
        with st.spinner("🔽 Downloading model from Hugging Face..."):
            response = requests.get(MODEL_URL)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            st.success("✅ Model downloaded successfully!")

    # Load model
    return tf.keras.models.load_model(MODEL_PATH)

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

/* ---- Diya Glow ---- */
.diya {
  width:80px;height:80px;margin:10px auto;
  background:radial-gradient(circle,rgba(255,200,50,0.8) 10%,rgba(255,120,0,0.4) 40%,rgba(0,0,0,0) 70%);
  border-radius:50%;
  animation:glow 1.5s infinite alternate;
  filter:blur(6px);
}
@keyframes glow {
  0%{opacity:0.85;transform:scale(1);}
  50%{opacity:1;transform:scale(1.1);}
  100%{opacity:0.8;transform:scale(0.95);}
}

/* ---- FILE UPLOADER ---- */
[data-testid="stFileUploader"] section{
  background:linear-gradient(135deg,#c79a63,#f5deb3,#c79a63);
  color:#3b200a;border:3px dashed #8b4513;border-radius:18px;
  padding:1.2rem;
  box-shadow:0 0 15px rgba(255,200,100,0.4);
  transition:all .3s ease-in-out;
}
[data-testid="stFileUploader"] section:hover{
  background:linear-gradient(135deg,#d4a76a,#ffebc1,#c79a63);
  box-shadow:0 0 25px rgba(255,215,100,.7);
  border-color:#b87333;
  transform:scale(1.02);
}
[data-testid="stFileUploader"] label{
  color:#4b2b16!important;
  font-weight:600;
  font-size:1.05rem;
}

/* ---- PREDICTION CARDS ---- */
.prediction-card{
  background:linear-gradient(145deg,#f7e3b3,#e2c28d);
  border-radius:15px;padding:1rem;text-align:center;
  box-shadow:0 2px 10px rgba(0,0,0,.25);
  transition:transform .3s,box-shadow .3s;
}
.prediction-card:hover{
  transform:scale(1.05);
  box-shadow:0 4px 20px rgba(255,180,80,.6);
}
.stProgress>div>div>div>div{
  background-image:linear-gradient(to right,#ff9f0f,#b87333);
}
.result-text {
  color:#ffef9f;
  font-size:1.4rem;
  font-weight:600;
  text-shadow:0 0 12px rgba(255,230,120,0.9);
}
.idol-info{
  margin-top:18px;
  border-radius:12px;
  padding:1rem;
  text-align:center;
  box-shadow:0 4px 15px rgba(0,0,0,0.25);
  color:#2b1a09;
  font-size:1.05rem;
  font-weight:500;
  border:2px solid rgba(255,215,128,0.7);
}

/* ---- SIDEBAR ANCIENT THEME ---- */
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
section[data-testid="stSidebar"] .stRadio label div {
  background: rgba(255,245,200,0.1);
  padding: 0.3rem 0.8rem;
  border-radius: 8px;
  transition: all 0.2s ease-in-out;
}
section[data-testid="stSidebar"] .stRadio label:hover div {
  background: rgba(255,230,150,0.2);
  transform: scale(1.05);
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
  <div class='diya' style="margin-bottom:12px;"></div>
  <h1 style="margin:0;color:#fff8dc;text-shadow:0 0 15px rgba(255,210,100,0.9);
             font-size:2.6rem;letter-spacing:1px;">
    🕉️ Divine Idol Classifier
  </h1>
  <p style="margin-top:0.4rem;color:#ffeaa7;font-size:1.2rem;
            text-shadow:0 0 10px rgba(255,240,180,0.9);">
    Enlightened by AI — Discover Vishnu, Mahadev, or Sursundari 🔱
  </p>
  <div style="height:4px;width:60%;margin:15px auto;
              background:linear-gradient(90deg,rgba(255,215,128,0.8),
              rgba(255,230,180,0.3),rgba(255,215,128,0.8));
              border-radius:50px;"></div>
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
        c1, c2 = st.columns([1, 1.3])
        with c1:
            img = Image.open(file).convert("RGB")
            st.image(img, caption="Uploaded Idol", use_container_width=True)
        with c2:
            st.subheader("🔮 Prediction Results")
            probs, top3 = predict(img)
            main_label, main_conf = top3[0]
            key = main_label.lower().strip()
            info = idol_info.get(key, {"text": "No information available.",
                                       "bg": "linear-gradient(145deg,#f8e7c1,#e7c991)"})
            st.markdown(f"<p class='result-text'>✨ Most likely: <b>{main_label}</b> ({main_conf*100:.2f}%)</p>", unsafe_allow_html=True)
            st.markdown(f"<div class='idol-info' style='background:{info['bg']};'><b>{main_label}</b>: {info['text']}</div>", unsafe_allow_html=True)
            st.markdown("### 🪷 Confidence Breakdown")
            for name, prob in zip(class_names, probs):
                st.write(f"**{name}**: {prob*100:.2f}%")
                st.progress(float(prob))
            st.markdown("### 🏛️ Top 3 Predictions")
            c1, c2, c3 = st.columns(3)
            for (lbl, conf), col in zip(top3, [c1, c2, c3]):
                col.markdown(f"<div class='prediction-card'><h4>{lbl}</h4><p>{conf*100:.2f}%</p></div>", unsafe_allow_html=True)

else:
    c1, c2 = st.columns(2)
    with c1:
        f1 = st.file_uploader("📷 Upload Image 1", type=["jpg","jpeg","png"], key="1")
    with c2:
        f2 = st.file_uploader("📸 Upload Image 2", type=["jpg","jpeg","png"], key="2")
    if f1 and f2:
        c1, c2 = st.columns(2)
        with c1:
            img1 = Image.open(f1).convert("RGB")
            st.image(img1, caption="Image 1", use_container_width=True)
            p1, t1 = predict(img1)
            label1, conf1 = t1[0]
            st.markdown(f"<p class='result-text'>✨ {label1} ({conf1*100:.2f}%)</p>", unsafe_allow_html=True)
        with c2:
            img2 = Image.open(f2).convert("RGB")
            st.image(img2, caption="Image 2", use_container_width=True)
            p2, t2 = predict(img2)
            label2, conf2 = t2[0]
            st.markdown(f"<p class='result-text'>✨ {label2} ({conf2*100:.2f}%)</p>", unsafe_allow_html=True)
        st.markdown("### 📊 Confidence Comparison")
        for i, name in enumerate(class_names):
            st.write(f"**{name}**")
            c1, c2 = st.columns(2)
            c1.progress(float(p1[i])); c2.progress(float(p2[i]))

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
