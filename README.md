<div align="center">

# 🛕 Classification and Identification of Idols
### Deep Learning-Based Hindu Idol Recognition using EfficientNetB0 & Streamlit

<p align="center">
<img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python">
<img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow">
<img src="https://img.shields.io/badge/Streamlit-Web_App-FF4B4B?style=for-the-badge&logo=streamlit">
<img src="https://img.shields.io/badge/Model-EfficientNetB0-success?style=for-the-badge">
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge">
</p>

*A Deep Learning application that identifies and classifies Hindu idols from uploaded images using Transfer Learning with EfficientNetB0.*

</div>

---

# 📖 Overview

This project is an AI-powered image classification system designed to identify Hindu idols from uploaded images.

The application leverages **Transfer Learning** using **EfficientNetB0** and provides an elegant **Streamlit-based web interface** for real-time predictions.

Apart from identifying the idol, the application also displays confidence scores, probability distribution, and allows users to compare two idol images simultaneously.

---

# ✨ Features

- 🛕 Classifies Hindu idols from uploaded images
- 🧠 EfficientNetB0 Transfer Learning model
- 📷 Drag & Drop Image Upload
- 📊 Confidence Score Visualization
- 🥇 Top-3 Prediction Ranking
- ⚖️ Compare Two Images Mode
- 🎨 Temple-inspired Modern UI
- ⚡ Real-Time Predictions
- 📈 Confidence Comparison Charts
- 💾 Trained Model Saved in `.keras` Format

---

# 🏛️ Idol Categories

The model currently classifies images into the following categories:

- 🔱 Lord Shiva (Mahadev)
- 🪷 Lord Vishnu
- 🌸 Sursundari

---

# 🖥️ Application Screenshots

## 🏠 Home Page


<img src="/screenshots/home-page.png"/>




---

## 🔍 Single Image Prediction

> Displays

- Uploaded Image
- Predicted Idol
- Confidence Score
- Top 3 Predictions
- Probability Distribution

```text
screenshots/prediction.png
```

---

## ⚖️ Compare Two Images

Users can upload two different idol images and compare:

- Predicted Classes
- Confidence Scores
- Confidence Comparison Graph

```text
screenshots/compare.png
```

---

# 🧠 Model Architecture

The project uses **EfficientNetB0** pretrained on ImageNet.

```
Input Image
      │
      ▼
Resize (224 × 224)
      │
      ▼
Data Augmentation
      │
      ▼
EfficientNetB0
      │
      ▼
Global Average Pooling
      │
      ▼
Dropout
      │
      ▼
Dense Layer
      │
      ▼
Softmax Output
      │
      ▼
Predicted Idol
```

---

# 📊 Training Strategy

The model was trained using:

- Transfer Learning
- EfficientNetB0 Backbone
- ImageNet Pretrained Weights
- Data Augmentation
- Class Weight Balancing
- Early Stopping
- Validation Monitoring

---

# 📈 Model Information

| Property | Value |
|----------|-------|
| Model | EfficientNetB0 |
| Framework | TensorFlow / Keras |
| Frontend | Streamlit |
| Input Size | 224 × 224 |
| Number of Classes | 3 |
| Output | Softmax Probabilities |
| Transfer Learning | ✅ |
| Confidence Scores | ✅ |

---

# 🛠️ Tech Stack

## Programming

- Python

## Deep Learning

- TensorFlow
- Keras
- EfficientNetB0

## Web Framework

- Streamlit

## Image Processing

- Pillow

## Data Processing

- NumPy
- Scikit-learn

## Visualization

- Matplotlib

---

# 📂 Project Structure

```
classification-and-identification-of-idols/

│── app.py
│── model.py
│── test_model.py
│── augment_dataset.py
│── class_names.json
│── idols_classifier_fixed.keras
│── requirements.txt
│── README.md

├── dataset/
│     ├── train/
│     └── validation/

├── screenshots/

├── idols_accuracy_fixed.png
└── idols_loss_fixed.png
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/Harshal1804/classification-and-identification-of-idols.git
```

Move inside the project

```bash
cd classification-and-identification-of-idols
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

The Streamlit application will automatically open in your browser.

---

# 🏋️ Training the Model

To retrain the model:

```bash
python model.py
```

The script will:

- Load dataset
- Apply preprocessing
- Train EfficientNetB0
- Save trained model
- Generate accuracy & loss graphs

---

# 🔍 Testing the Model

Run

```bash
python test_model.py
```

The application predicts the uploaded image and displays:

- Predicted Idol
- Confidence Score
- Top-3 Predictions
- Probability Distribution

---

# 📸 Prediction Workflow

```
Upload Image
      │
      ▼
Image Preprocessing
      │
      ▼
EfficientNetB0 Prediction
      │
      ▼
Softmax Probability
      │
      ▼
Prediction Result
      │
      ├── Confidence Score
      ├── Top 3 Predictions
      └── Probability Breakdown
```

---

# 🚀 Future Improvements

- Add more idol categories
- Support multiple idols in one image
- Object Detection (YOLO)
- Mobile Application
- REST API Deployment
- Cloud Deployment
- Explainable AI (Grad-CAM)
- Multi-language Support

---

# 🎯 Applications

- Cultural Heritage Preservation
- Museum Digitization
- Archaeological Research
- Religious Artifact Documentation
- Educational Platforms
- AI-assisted Idol Identification

---

# 👨‍💻 Author

## Harshal Shinde

Computer Engineering Student

MIT ADT University, Pune

🔗 GitHub: **https://github.com/Harshal1804**

---

# 🙏 Acknowledgements

- TensorFlow
- Keras
- Streamlit
- EfficientNet Research
- ImageNet Dataset

---

# ⭐ Show Your Support

If you found this project interesting, consider giving it a ⭐ on GitHub!

---

# 📜 License

This project is developed for educational and research purposes.

Feel free to use and modify it with proper attribution.
