# 🚗 Accident Detection System using CNN and ResNet50 in 3D Architecture

## 📌 Overview

This project is a deep learning-based **Accident Detection System** designed to identify vehicular accidents in real-time from video data. Leveraging **3D Convolutional Neural Networks (3D CNNs)** and a pretrained **ResNet50** architecture, the system processes sequences of frames to learn both spatial and temporal features — enabling it to detect sudden movements, collisions, and anomalies typical of road accidents.

---

## 🎯 Objectives

- Detect accidents from dashcam, surveillance, or traffic footage.
- Utilize **3D CNN** for spatiotemporal feature extraction.
- Integrate **ResNet50** for deep residual learning and accurate classification.
- Enable near real-time predictions for safety monitoring systems.


---

## FIND THE DATA SET IN THIS GOOGLE DRIVE 

https://drive.google.com/drive/folders/18R_-TVD0jNkAKGhISgW84VfFAXyn_HTe?usp=drive_link

---
---

## 🧰 Technologies & Tools

- **Python**
- **TensorFlow / Keras**
- **OpenCV** (for frame extraction and video preprocessing)
- **ResNet50** pretrained on ImageNet (for transfer learning)
- **3D CNN layers** for temporal feature learning
- (Optional) **Streamlit** or **Flask** for UI-based inference demo

---

## 📷 How It Works

1. **Frame Extraction**  
   Input video is divided into sequences of frames (clips), typically 16–32 frames long.

2. **Preprocessing**  
   Each clip is resized, normalized, and formatted into a tensor for model input.

3. **Model Architecture**  
   - **ResNet50** is used as the base feature extractor.
   - Extended with **3D convolutional layers** to capture motion over time.
   - Final dense layers classify whether an accident has occurred.

4. **Prediction**  
   The model returns a binary label: `Accident` or `Normal`.

---

## 🚀 Getting Started

### Clone the repository

```bash
# Clone the repository
git clone https://github.com/abhinai2244/ACCIDENT-DETECTION-SYSTEM-USING-CNN-AND-RESNET50.git

# Navigate to the project directory
cd ACCIDENT-DETECTION-SYSTEM-USING-CNN-AND-RESNET50

# Install dependencies
pip install -r requirements.txt

