# streamlit_app.py

import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import tempfile
import time
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode, RTCConfiguration
import matplotlib.pyplot as plt

# ==============================
# Configuration and Setup
# ==============================

# Set page configuration
st.set_page_config(
    page_title="Real-Time Object Detection",
    layout="wide",
)

# RTC Configuration for WebRTC (optional, can be customized)
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Paths for YOLOv5 model and labels
YOLOV5N_MODEL_PATH = "yolov5n.pt"
LABEL_FILE = "coco.names"

# Paths for SSD MobileNet model files
CONFIG_FILE = "C:/Users/KIIT/Music/objdetec/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
FROZEN_MODEL = "C:/Users/KIIT/Music/objdetec/frozen_inference_graph.pb"
LABELS_FILE = "C:/Users/KIIT/Music/objdetec/label.txt"

# ==============================
# Load Labels and Models
# ==============================

@st.cache_resource
def load_labels(label_path):
    with open(label_path, 'rt') as fpt:
        class_labels = fpt.read().rstrip('\n').split('\n')
    return class_labels

@st.cache_resource
def load_yolov5_model(model_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    model.eval()
    return model

@st.cache_resource
def load_ssd_mobilenet_model(config_file, frozen_model):
    model = cv2.dnn_DetectionModel(frozen_model, config_file)
    model.setInputSize(320, 320)
    model.setInputScale(1.0 / 127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)
    return model

yolov5_labels = load_labels(LABEL_FILE)
ssd_labels = load_labels(LABELS_FILE)
yolov5_model = load_yolov5_model(YOLOV5N_MODEL_PATH)
ssd_model = load_ssd_mobilenet_model(CONFIG_FILE, FROZEN_MODEL)

# ==============================
# Streamlit Sidebar
# ==============================

st.sidebar.title("Settings")

# Confidence threshold slider
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.3,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Adjust the confidence threshold for object detection."
)

# Mode selection
mode = st.sidebar.radio("Select Mode", ("Webcam", "Upload Video"))

# Model selection
selected_model = st.sidebar.selectbox("Select Detection Model", ("YOLOv5", "SSD MobileNet"))

# ==============================
# Helper Functions
# ==============================

def detect_objects_yolov5(frame, model, labels, conf_thresh=0.5):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img)
    detection_boxes = results.xyxy[0].cpu().numpy()
    output_frame = frame.copy()

    for *box, confidence, class_id in detection_boxes:
        if confidence >= conf_thresh:
            class_id = int(class_id)
            label = f"{labels[class_id]}: {int(confidence * 100)}%"
            cv2.rectangle(output_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(0, 255, 0), thickness=2)
            cv2.putText(output_frame, label, (int(box[0]), int(box[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return output_frame

def detect_objects_ssd(frame, model, labels, conf_thresh=0.5):
    class_indices, confidences, boxes = model.detect(frame, confThreshold=conf_thresh)
    output_frame = frame.copy()

    for class_index, confidence, box in zip(class_indices.flatten(), confidences.flatten(), boxes):
        if class_index <= len(labels):
            label = f"{labels[class_index - 1]}: {int(confidence * 100)}%"
            cv2.rectangle(output_frame, box, color=(255, 0, 0), thickness=2)
            cv2.putText(output_frame, label, (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return output_frame

# ==============================
# Real-Time Object Detection with Webcam
# ==============================

if mode == "Webcam":
    st.header("📷 Real-Time Object Detection using Webcam")

    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.yolov5_model = yolov5_model
            self.ssd_model = ssd_model
            self.yolov5_labels = yolov5_labels
            self.ssd_labels = ssd_labels
            self.conf_threshold = confidence_threshold
            self.selected_model = selected_model

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")

            if self.selected_model == "YOLOv5":
                output_img = detect_objects_yolov5(img, self.yolov5_model, self.yolov5_labels, conf_thresh=self.conf_threshold)
            else:
                output_img = detect_objects_ssd(img, self.ssd_model, self.ssd_labels, conf_thresh=self.conf_threshold)

            return output_img

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )

# ==============================
# Video Upload and Analysis
# ==============================

elif mode == "Upload Video":
    st.header("📂 Upload and Analyze Video")

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        screen_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        screen_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        video_placeholder = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if selected_model == "YOLOv5":
                output_frame = detect_objects_yolov5(frame, yolov5_model, yolov5_labels, conf_thresh=confidence_threshold)
            else:
                output_frame = detect_objects_ssd(frame, ssd_model, ssd_labels, conf_thresh=confidence_threshold)

            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(output_frame, channels="RGB", use_column_width=True)

            time.sleep(1 / fps)

        cap.release()
        st.success("Video processing completed.")

# ==============================
# Additional Information
# ==============================

st.sidebar.markdown("""
---
**Developed by Akashdip Saha**
""")
