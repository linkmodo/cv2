import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import av

# ---------------------
# Page Configuration & Styling
# ---------------------
st.set_page_config(page_title="Deep Learning Face Detection Model using OpenCV", layout="wide")
st.markdown("""
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
""", unsafe_allow_html=True)

st.markdown("""
<h1 style="text-align: center;
    background: -webkit-linear-gradient(75deg, orange, yellow);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    color: black;">
Deep Learning Based Face Detection
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<h3 style="text-align: center; color: white; font-size: 20px;">
This app detects face(s) in images, videos, and webcam streams using OpenCV's deep learning model.<br>
*This app does NOT save any user data after exiting*
</h3>
""", unsafe_allow_html=True)

# ---------------------
# Load DNN Model
# ---------------------
@st.cache_resource()
def load_model():
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

net = load_model()

# ---------------------
# Webcam Face Detection
# ---------------------
st.markdown("<h2 style='text-align: center; color: white;'>Webcam Face Detection</h2>", unsafe_allow_html=True)

conf_threshold_webcam = st.slider("Confidence Threshold (Webcam)", 0.0, 1.0, 0.5, 0.01)
box_color_webcam_hex = st.color_picker("Bounding Box Color (Webcam)", "#00FF00")
thickness_webcam = st.slider("Bounding Box Thickness (Webcam)", 1, 10, 4)
show_labels = st.checkbox("Show Confidence Score Labels", value=True)

resolution = st.selectbox("Webcam Resolution", options=["640x480", "1280x720", "1920x1080"], index=0)
width, height = map(int, resolution.split('x'))

# Convert hex to BGR
box_color_webcam = tuple(int(box_color_webcam_hex[i:i+2], 16) for i in (1, 3, 5))
box_color_webcam = (box_color_webcam[2], box_color_webcam[1], box_color_webcam[0])

# Update session state
st.session_state.conf_threshold_webcam = conf_threshold_webcam
st.session_state.box_color_webcam = box_color_webcam
st.session_state.thickness_webcam = thickness_webcam
st.session_state.show_labels = show_labels

rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class FaceDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        threshold = st.session_state.get("conf_threshold_webcam", 0.5)
        box_color = st.session_state.get("box_color_webcam", (0, 255, 0))
        thickness = st.session_state.get("thickness_webcam", 2)
        show_labels = st.session_state.get("show_labels", True)

        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()

        img_h, img_w = img.shape[:2]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > threshold:
                x1 = int(detections[0, 0, i, 3] * img_w)
                y1 = int(detections[0, 0, i, 4] * img_h)
                x2 = int(detections[0, 0, i, 5] * img_w)
                y2 = int(detections[0, 0, i, 6] * img_h)

                x1 = max(0, min(x1, img_w - 1))
                y1 = max(0, min(y1, img_h - 1))
                x2 = max(0, min(x2, img_w - 1))
                y2 = max(0, min(y2, img_h - 1))

                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, thickness, cv2.LINE_8)

                if show_labels:
                    label = f"{confidence:.2f}"
                    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    y1 = max(y1, label_size[1])
                    cv2.rectangle(img, (x1, y1 - label_size[1]), (x1 + label_size[0], y1), box_color, cv2.FILLED)
                    cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="face-detection",
    mode=WebRtcMode.SENDRECV,
    video_transformer_factory=FaceDetectionTransformer,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": {"width": width, "height": height}, "audio": False},
    async_processing=True,
)

st.info("If the webcam does not start automatically, check your browser permissions and click 'START'. Allow camera access when prompted.")

# ---------------------
# Footer
# ---------------------
st.markdown("""
<hr>
<p style="text-align: center; color: gray;">
Built by Li Fan 2025-03-01 | Powered by OpenCV & Streamlit
</p>
""", unsafe_allow_html=True)
