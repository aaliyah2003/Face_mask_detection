# 😷 Face Mask Detection using Deep Learning and OpenCV

This project performs real-time face mask detection using a webcam. It uses a **Convolutional Neural Network (CNN)** based on **MobileNetV2** to classify faces as either **with mask** or **without mask**, and integrates **OpenCV** for face detection and live video streaming.

## 📌 Features

- Real-time face mask detection via webcam
- Uses MobileNetV2 for lightweight and accurate classification
- High accuracy on custom dataset
- Face detection powered by OpenCV's DNN module
- Custom training script to train on your own dataset

## ⚙️ Installation and Setup

### ✅ Prerequisites

Ensure you have **Python 3.7** installed. It's recommended to use a virtual environment.

### 🔧 Step 1: Clone the Repository

git clone <https://github.com/aaliyah2003/Face_mask_detection.git>

## 🧪 Step 2: Create and Activate Virtual Environment

### On Windows:
python -m venv tf37_env
tf37_env\Scripts\activate

### On macOS/Linux:
python3 -m venv tf37_env
source tf37_env/bin/activate

## 📦 Step 3: Install Required Dependencies

pip install --upgrade pip
pip install tensorflow==2.10.0 keras==2.10.0
pip install opencv-python imutils matplotlib scikit-learn pillow

### You can also create a requirements.txt file with:
tensorflow==2.10.0
keras==2.10.0
opencv-python
imutils
matplotlib
scikit-learn
pillow
## And install with:
pip install -r requirements.txt

## 🏋️‍♂️ Training Your Own Model
## 1. Run the Training Script
python training_mask.py

## 🎥 Real-Time Detection (Video Stream)
✅ Update the Model Path (if needed)
Open detect_mask_video.py and ensure the model path is correct:
model = load_model("mask_detector.model") 

▶️ Run the Script
python realtime_detection.py

# 📈 Model Performance
Final Training & Validation Accuracy:
| Epoch | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
| ----- | ----------------- | ------------------- | ------------- | --------------- |
| 18    | 0.9921            | 0.9935              | 0.0261        | 0.0265          |
| 19    | 0.9898            | 0.9935              | 0.0267        | 0.0269          |
| 20    | **0.9941**        | **0.9935**          | **0.0236**    | **0.0254**      |


Classification Report on Test Set:
| Class        | Precision | Recall   | F1-Score | Support |
| ------------ | --------- | -------- | -------- | ------- |
| With Mask    | 0.99      | 0.99     | 0.99     | 383     |
| Without Mask | 0.99      | 0.99     | 0.99     | 384     |
| **Overall**  | **0.99**  | **0.99** | **0.99** | **767** |

## 🧠 Technologies Used
Python 3.7
TensorFlow 2.10.0
Keras 2.10.0
OpenCV (for face detection)
scikit-learn (for label encoding and metrics)
imutils (image preprocessing)

## 🙌 Acknowledgments
1. Face detection model from OpenCV’s DNN module
2. MobileNetV2 architecture from Keras Applications
3. Inspired by the real-world need for mask compliance detection during COVID-19

🛠️ Author
Aaliyah Shaikh
<https://github.com/aaliyah2003>
