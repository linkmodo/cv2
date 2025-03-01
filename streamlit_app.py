import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64

# Load logo and display it at the top
# logo = "logo.png"  # Ensure you have a logo file in the same directory
# st.image(logo, use_column_width=True)

# Create application title and file uploader widget.
st.title("OpenCV Deep Learning based Face Detection")
st.write("Detect faces in images using OpenCV's deep learning model.")

img_file_buffer = st.file_uploader("Choose a file", type=['jpg', 'jpeg', 'png'])

# Function for detecting faces in an image.
def detectFaceOpenCVDnn(net, frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    return detections

# Function for annotating the image with bounding boxes for each detected face.
def process_detections(frame, detections, conf_threshold=0.5, box_color=(0, 255, 0), thickness=2):
    bboxes = []
    frame_h, frame_w = frame.shape[:2]
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_w)
            y1 = int(detections[0, 0, i, 4] * frame_h)
            x2 = int(detections[0, 0, i, 5] * frame_w)
            y2 = int(detections[0, 0, i, 6] * frame_h)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness, cv2.LINE_8)
    return frame, bboxes

# Load the DNN model.
@st.cache_resource()
def load_model():
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

# Function to generate a download link for output file.
def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

net = load_model()

if img_file_buffer is not None:
    raw_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    
    placeholders = st.columns(2)
    placeholders[0].image(image, channels='BGR')
    placeholders[0].text("Input Image")
    
    # Adjustable Parameters
    conf_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, step=.01, value=0.5)
    box_color = st.color_picker("Bounding Box Color", "#00FF00")  # Default green
    thickness = st.slider("Bounding Box Thickness", 1, 10, 2)
    
    # Convert hex color to BGR
    box_color = tuple(int(box_color[i:i+2], 16) for i in (1, 3, 5))
    box_color = (box_color[2], box_color[1], box_color[0])
    
    detections = detectFaceOpenCVDnn(net, image)
    out_image, _ = process_detections(image, detections, conf_threshold, box_color, thickness)
    
    placeholders[1].image(out_image, channels='BGR')
    placeholders[1].text("Output Image")
    
    out_image = Image.fromarray(out_image[:, :, ::-1])
    st.markdown(get_image_download_link(out_image, "face_output.jpg", 'Download Output Image'), unsafe_allow_html=True)
    
# Footer with credits
st.markdown("""
---
**Built by Li Fan** 2025-03-01 | Powered by OpenCV & Streamlit
""")
