import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import tempfile
import os

# Load logo and display it at the top
# logo = "logo.png"  # Ensure you have a logo file in the same directory
# st.image(logo, use_column_width=True)

# Create application title and file uploader widget.
st.title("Deep Learning based Face Detection Using OpenCV")
st.write("Detect faces in images and videos using OpenCV's deep learning model. Upload an image or video to start.")

img_file_buffer = st.file_uploader("Choose an image file to be analyzed", type=['jpg', 'jpeg', 'png'])
video_file_buffer = st.file_uploader("Choose a video file to be analyzed", type=['mp4', 'avi', 'mov'])

# Function for detecting faces in an image.
def detectFaceOpenCVDnn(net, frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    return detections

# Function for annotating the image with bounding boxes for each detected face.
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

# Function to rotate an image
def rotate_image(image, angle):
    if angle != 0:
        image = cv2.rotate(image, angle)
    return image

# Load the DNN model.
@st.cache_resource()
def load_model():
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

net = load_model()

# Image Processing
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
    rotation_angle = st.selectbox("Rotate Image", [0, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180], format_func=lambda x: "None" if x == 0 else "90° CW" if x == cv2.ROTATE_90_CLOCKWISE else "90° CCW" if x == cv2.ROTATE_90_COUNTERCLOCKWISE else "180°")
    
    box_color = tuple(int(box_color[i:i+2], 16) for i in (1, 3, 5))
    box_color = (box_color[2], box_color[1], box_color[0])
    
    image = rotate_image(image, rotation_angle)
    detections = detectFaceOpenCVDnn(net, image)
    out_image = process_detections(image, detections, conf_threshold, box_color, thickness)
    
    placeholders[1].image(out_image, channels='BGR')
    placeholders[1].text("Output Image")
    
    out_image = Image.fromarray(out_image[:, :, ::-1])
    # Save the processed image to BytesIO
    buf = BytesIO()
    out_image.save(buf, format='JPEG')
    st.download_button("Download Processed Image", data=buf.getvalue(), file_name="processed_image.jpg", mime="image/jpeg")

# Video Processing
if video_file_buffer is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file_buffer.read())
    video_path = tfile.name
    
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Get original video dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    conf_threshold = st.slider("Confidence Threshold for Video", 0.0, 1.0, 0.5, 0.01)
    box_color = st.color_picker("Bounding Box Color for Video", "#00FF00")
    # Convert hex color to BGR
    box_color = tuple(int(box_color[i:i+2], 16) for i in (1, 3, 5))
    box_color = (box_color[2], box_color[1], box_color[0])
    thickness = st.slider("Bounding Box Thickness for Video", 1, 10, 2)
    rotation_angle = st.selectbox("Rotate Video", [0, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180], format_func=lambda x: "None" if x == 0 else "90° CW" if x == cv2.ROTATE_90_CLOCKWISE else "90° CCW" if x == cv2.ROTATE_90_COUNTERCLOCKWISE else "180°")
    
    # Adjust output dimensions based on rotation
    out_width = width
    out_height = height
    if rotation_angle in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
        out_width = height
        out_height = width
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    progress_text = st.empty()
    progress_bar = st.progress(0)
    stframe = st.empty()
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        current_frame += 1
        progress = int((current_frame / frame_count) * 100)
        progress_text.text(f"Processing video: {progress}%")
        progress_bar.progress(progress)
        
        frame = rotate_image(frame, rotation_angle)
        detections = detectFaceOpenCVDnn(net, frame)
        frame = process_detections(frame, detections, conf_threshold, box_color, thickness)
        out.write(frame)
        stframe.image(frame, channels="BGR")
    
    cap.release()
    out.release()
    
    progress_text.text("Processing complete!")
    progress_bar.progress(100)
    
    # Clean up temporary input file
    try:
        os.unlink(tfile.name)
    except:
        pass
    
    with open(output_path, "rb") as f:
        st.download_button("Download Processed Video", f, file_name="processed_video.mp4", mime="video/mp4")
    
    # Clean up output file
    try:
        os.unlink(output_path)
    except:
        pass

# Footer with credits
st.markdown("""
---
**Built by Li Fan** 2025-03-01 | Powered by OpenCV & Streamlit
""")
