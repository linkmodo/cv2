import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import tempfile
import os

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
# Define Face Detection Functions
# ---------------------
def detectFaceOpenCVDnn(net, frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    return detections

def process_detections(frame, detections, conf_threshold=0.5, box_color=(0, 255, 0), thickness=2):
    frame_h, frame_w = frame.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_w)
            y1 = int(detections[0, 0, i, 4] * frame_h)
            x2 = int(detections[0, 0, i, 5] * frame_w)
            y2 = int(detections[0, 0, i, 6] * frame_h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness, cv2.LINE_8)
    return frame

def rotate_image(image, angle):
    if angle is None:
        return image
    if angle == cv2.ROTATE_90_CLOCKWISE:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == cv2.ROTATE_180:
        return cv2.rotate(image, cv2.ROTATE_180)
    else:
        return image

# ---------------------
# Image Processing Section
# ---------------------
img_file_buffer = st.file_uploader("Upload an image file with face(s) in it to be analyzed", type=['jpg', 'jpeg', 'png'])
if img_file_buffer is not None:
    raw_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    
    col1, col2 = st.columns(2)
    col1.image(image, channels='BGR', caption="Input Image")
    
    conf_threshold_img = st.slider("Confidence Threshold (Image)", 0.0, 1.0, 0.5, 0.01)
    box_color_hex = st.color_picker("Bounding Box Color (Image)", "#00FF00")
    thickness_img = st.slider("Bounding Box Thickness (Image)", 1, 10, 4)
    # Convert hex to BGR tuple
    box_color_img = tuple(int(box_color_hex[i:i+2], 16) for i in (1, 3, 5))
    box_color_img = (box_color_img[2], box_color_img[1], box_color_img[0])
    
    # Image rotation controls
    if 'rotation_angle_image' not in st.session_state:
        st.session_state.rotation_angle_image = None

    r1, r2, r3, r4 = st.columns(4)
    if r1.button("Rotate 90° CW (Image)"):
        st.session_state.rotation_angle_image = cv2.ROTATE_90_CLOCKWISE
    if r2.button("Rotate 90° CCW (Image)"):
        st.session_state.rotation_angle_image = cv2.ROTATE_90_COUNTERCLOCKWISE
    if r3.button("Rotate 180° (Image)"):
        st.session_state.rotation_angle_image = cv2.ROTATE_180
    if r4.button("Reset Rotation (Image)"):
        st.session_state.rotation_angle_image = None

    rotated_image = rotate_image(image, st.session_state.rotation_angle_image)
    detections = detectFaceOpenCVDnn(net, rotated_image)
    out_image = process_detections(rotated_image.copy(), detections, conf_threshold_img, box_color_img, thickness_img)
    
    col2.image(out_image, channels='BGR', caption="Output Image")
    out_image_pil = Image.fromarray(cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    out_image_pil.save(buf, format='JPEG')
    st.download_button("Download Processed Image", data=buf.getvalue(), file_name="processed_image.jpg", mime="image/jpeg")


# ---------------------
# Footer
# ---------------------
st.markdown("""
<hr>
<p style="text-align: center; color: gray;">
Built by Li Fan 2025-03-01 | Powered by OpenCV & Streamlit
</p>
""", unsafe_allow_html=True)
