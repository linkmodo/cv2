import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import tempfile
import os

# Import for webcam streaming using streamlit-webrtc
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# ---------------------
# Page Configuration & Styling
# ---------------------
st.set_page_config(page_title="Deep Learning Face Detection", layout="wide")
st.markdown(
    """
    <style>
    .block-container {
        max-width: 900px;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .stApp {
        background-image: url("https://raw.githubusercontent.com/linkmodo/cv2/main/theatre-4981934.jpg");
        background-size: cover;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("""
<h1 style="text-align: center;
    background: -webkit-linear-gradient(45deg, orange, yellow);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    color: black;">
Deep Learning Based Face Detection
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<h3 style="text-align: center; color: white; font-size: 20px;">
This application detects face(s) in images, videos, or webcam streams using OpenCV's deep learning model.<br>
<strong>No data is saved after exiting this page</strong>
</h3>
""", unsafe_allow_html=True)

# ---------------------
# Load the Face Detection Model
# ---------------------
@st.cache_resource()
def load_model():
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

net = load_model()

# ---------------------
# Define Face Detection Functions
# ---------------------
def detect_face(net, frame):
    # Preprocess the image: resize to 300x300, mean subtraction, etc.
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=(300, 300), 
                                 mean=(104, 117, 123), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()
    return detections

def process_detections(frame, detections, threshold=0.5, color=(0, 255, 0), thickness=2):
    (h, w) = frame.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, thickness)
    return frame

# ---------------------
# Define the VideoTransformer for streamlit-webrtc
# ---------------------
class FaceDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert the frame from streamlit-webrtc to a NumPy array (BGR format)
        img = frame.to_ndarray(format="bgr24")
        detections = detect_face(net, img)
        processed_img = process_detections(img, detections, threshold=0.5, color=(0, 255, 0), thickness=2)
        return processed_img

# ---------------------
# RTC Configuration for WebRTC ICE Servers
# ---------------------
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ---------------------
# Streamlit App Layout
# ---------------------
st.title("Webcam Face Detection")
st.write("Please allow camera access. For non-localhost deployments, ensure you use HTTPS.")

# ---------------------
# Start the Webcam Stream using streamlit-webrtc
# ---------------------
webrtc_streamer(
    key="face-detection",
    video_transformer_factory=FaceDetectionTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    async_transform=True  # Enable asynchronous frame processing for smoother display.
)

# ---------------------
# Footer
# ---------------------
st.markdown("""
<hr>
<p style="text-align: center; color: gray;">
Built by Li Fan 2025-03-01 | Powered by OpenCV & Streamlit
</p>
""", unsafe_allow_html=True)
