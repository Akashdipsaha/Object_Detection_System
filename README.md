# Real-Time Object Detection Streamlit App

This Streamlit application provides real-time object detection capabilities using two different models: YOLOv5 and SSD MobileNet. The app supports both webcam-based real-time detection and offline video file analysis.

## Features

- **Real-Time Object Detection**: Detect objects in real-time using your webcam.
- **Upload and Analyze Video**: Upload a video file (`mp4`, `avi`, `mov`) and perform object detection frame-by-frame.
- **Multiple Models**: Choose between YOLOv5 (small version) and SSD MobileNet for object detection.
- **Adjustable Confidence Threshold**: Customize the confidence threshold for object detection to fine-tune results.

## Setup and Installation

### 1. Clone the Repository
```
git clone https://github.com/your-username/real-time-object-detection-app.git
cd real-time-object-detection-app
```
### 2. Install Dependencies
Make sure you have Python 3.8 or higher installed. Then, install the required Python libraries:

```
pip install streamlit opencv-python-headless torch torchvision torchaudio streamlit-webrtc matplotlib
```

### 3. Download the Required Models and Files
YOLOv5 Model: Download the YOLOv5n model (yolov5n.pt) from Ultralytics YOLOv5 repository.

SSD MobileNet Model: Download the SSD MobileNet configuration and frozen model files:
```
ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
frozen_inference_graph.pb
COCO Labels: Download the coco.names file containing the class labels for YOLOv5.
```
Place these files in the appropriate paths specified in the streamlit_app.py file or update the paths accordingly.

### 4. Run the App
Execute the following command to run the Streamlit app:

```
streamlit run streamlit_app.py
```
Usage
Select the Mode: Choose between "Webcam" and "Upload Video" modes from the sidebar.


Webcam Mode: Allows real-time detection using your webcam.
Upload Video Mode: Allows you to upload a video file for analysis.
Select the Detection Model: Choose between "YOLOv5" and "SSD MobileNet" for object detection.

Adjust Confidence Threshold: Use the slider to set the desired confidence threshold for detection.

### Start Detection:

In "Webcam Mode", the app will start capturing video from your webcam and display detected objects in real-time.
In "Upload Video Mode", upload a video file, and the app will process each frame, displaying detected objects.
Requirements
Python 3.8 or higher
Webcam (for real-time detection)
Internet connection (for downloading models and libraries)
Credits
Developed by: Akashdip Saha
YOLOv5: Ultralytics YOLOv5
SSD MobileNet: TensorFlow Model Zoo
License
This project is licensed under the MIT License - see the LICENSE file for details.


### Instructions for Customizing the README

1. **Replace `your-username`**: Change `your-username` in the `git clone` command to your GitHub username.
2. **Model Links**: Provide direct links to the models and files if necessary.
3. **Additional Information**: Add any additional information that is specific to your use c
