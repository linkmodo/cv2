import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import tempfile
import os

# Set page configuration to wide and restrict container width to 900px.
st.set_page_config(page_title="Deep Learning Face Detection", layout="wide")
st.markdown(
    """
    <style>
    .block-container {
        max-width: 900px;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and subtitle
st.markdown("""
<h1 style="
    text-align: center;
    background: -webkit-linear-gradient(45deg, orange, yellow);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    color: black; /* fallback color */
">
Deep Learning Based Face Detection
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<h3 style="
    text-align: center;
    color: white;
    font-size: 20px;
">
This application detects face(s) in images, videos, and via your live webcam using OpenCV's deep learning model.<br>
Start by uploading an image or video file, or use your webcam below.
</h3>
""", unsafe_allow_html=True)

# File uploaders for image and video.
img_file_buffer = st.file_uploader("Choose an image file with face(s) in it to be analyzed", type=['jpg', 'jpeg', 'png'])
video_file_buffer = st.file_uploader("Choose a video file with face(s) in it to be analyzed", type=['mp4', 'avi', 'mov'])

# Function for detecting faces using OpenCV's DNN.
def detectFaceOpenCVDnn(net, frame):
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    return detections

# Function for drawing bounding boxes on the frame.
def process_detections(frame, detections, conf_threshold=0.5, box_color=(0, 255, 0), thickness=4):
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

# Function to rotate an image using the specified angle.
def rotate_image(image, angle):
    if angle == cv2.ROTATE_90_CLOCKWISE:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == cv2.ROTATE_180:
        return cv2.rotate(image, cv2.ROTATE_180)
    else:
        return image

# Load the DNN model.
@st.cache_resource()
def load_model():
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net

net = load_model()

# ---------------------
# Image Processing
# ---------------------
if img_file_buffer is not None:
    raw_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    
    placeholders = st.columns(2)
    placeholders[0].image(image, channels='BGR', caption="Input Image")
    
    # Adjustable Parameters
    conf_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, step=0.01, value=0.5)
    box_color_hex = st.color_picker("Bounding Box Color", "#00FF00")
    thickness = st.slider("Bounding Box Thickness", 1, 10, 8)
    # Convert hex to BGR tuple.
    box_color = tuple(int(box_color_hex[i:i+2], 16) for i in (1, 3, 5))
    box_color = (box_color[2], box_color[1], box_color[0])
    
    # Rotation buttons using session state.
    if 'rotation_angle_image' not in st.session_state:
        st.session_state.rotation_angle_image = None

    col1, col2, col3, col4 = st.columns(4)
    if col1.button("Rotate 90° CW"):
        st.session_state.rotation_angle_image = cv2.ROTATE_90_CLOCKWISE
    if col2.button("Rotate 90° CCW"):
        st.session_state.rotation_angle_image = cv2.ROTATE_90_COUNTERCLOCKWISE
    if col3.button("Rotate 180°"):
        st.session_state.rotation_angle_image = cv2.ROTATE_180
    if col4.button("Reset Rotation"):
        st.session_state.rotation_angle_image = None

    # Apply the selected rotation.
    rotated_image = rotate_image(image, st.session_state.rotation_angle_image)
    detections = detectFaceOpenCVDnn(net, rotated_image)
    out_image = process_detections(rotated_image, detections, conf_threshold, box_color, thickness)
    
    placeholders[1].image(out_image, channels='BGR', caption="Output Image")
    
    out_image_pil = Image.fromarray(out_image[:, :, ::-1])
    buf = BytesIO()
    out_image_pil.save(buf, format='JPEG')
    st.download_button("Download Processed Image", data=buf.getvalue(), file_name="processed_image.jpg", mime="image/jpeg")

# ---------------------
# Video Processing
# ---------------------
if video_file_buffer is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file_buffer.read())
    video_path = tfile.name
    
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    conf_threshold_video = st.slider("Confidence Threshold for Video", 0.0, 1.0, 0.5, 0.01)
    box_color_video_hex = st.color_picker("Bounding Box Color for Video", "#00FF00")
    box_color_video = tuple(int(box_color_video_hex[i:i+2], 16) for i in (1, 3, 5))
    box_color_video = (box_color_video[2], box_color_video[1], box_color_video[0])
    thickness_video = st.slider("Bounding Box Thickness for Video", 1, 10, 8)
    
    # Video rotation buttons.
    if 'rotation_angle_video' not in st.session_state:
        st.session_state.rotation_angle_video = None

    col1, col2, col3, col4 = st.columns(4)
    if col1.button("Rotate 90° CW (Video)"):
        st.session_state.rotation_angle_video = cv2.ROTATE_90_CLOCKWISE
    if col2.button("Rotate 90° CCW (Video)"):
        st.session_state.rotation_angle_video = cv2.ROTATE_90_COUNTERCLOCKWISE
    if col3.button("Rotate 180° (Video)"):
        st.session_state.rotation_angle_video = cv2.ROTATE_180
    if col4.button("Reset Rotation (Video)"):
        st.session_state.rotation_angle_video = None

    rotation_angle_video = st.session_state.rotation_angle_video
    
    # Adjust output dimensions if a 90° rotation is selected.
    out_width = width
    out_height = height
    if rotation_angle_video in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
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
        
        frame = rotate_image(frame, rotation_angle_video)
        detections = detectFaceOpenCVDnn(net, frame)
        frame = process_detections(frame, detections, conf_threshold_video, box_color_video, thickness_video)
        out.write(frame)
        stframe.image(frame, channels="BGR")
    
    cap.release()
    out.release()
    
    progress_text.text("Processing complete!")
    progress_bar.progress(100)
    
    try:
        os.unlink(tfile.name)
    except Exception as e:
        st.error(f"Error deleting temporary file: {e}")
    
    with open(output_path, "rb") as f:
        st.download_button("Download Processed Video", f, file_name="processed_video.mp4", mime="video/mp4")
    
    try:
        os.unlink(output_path)
    except Exception as e:
        st.error(f"Error deleting temporary file: {e}")

# ---------------------
# Live Webcam Face Detection
# ---------------------
st.markdown("""
<h3 style="
    text-align: center;
    color: white;
    font-size: 20px;
">
Live Webcam Face Detection
</h3>
""", unsafe_allow_html=True)

if st.button("Start Webcam"):
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Unable to access webcam. Please check your camera permissions.")
        else:
            # Add webcam settings
            conf_threshold_webcam = st.slider(
                "Confidence Threshold (Webcam)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.01
            )
            
            box_color_webcam = (0, 255, 0)  # Default green
            thickness_webcam = 4
            
            stframe = st.empty()
            stop_button = st.button("Stop Webcam")
            
            while not stop_button:
                ret, frame = cap.read()
                if ret:
                    # Run face detection
                    detections = detectFaceOpenCVDnn(net, frame)
                    
                    # Process and annotate the frame
                    processed_frame = process_detections(
                        frame.copy(),
                        detections,
                        conf_threshold_webcam,
                        box_color_webcam,
                        thickness_webcam
                    )
                    
                    # Display the frame
                    stframe.image(processed_frame, channels="BGR")
                else:
                    st.error("Error reading from webcam")
                    break
            
            cap.release()
            stframe.empty()
            
    except Exception as e:
        st.error(f"An error occurred with the webcam: {str(e)}")
        if 'cap' in locals():
            cap.release()

# Footer
st.markdown("""
<hr>
<p style="text-align: center; color: gray;">
Built by Li Fan 2025-03-01 | Powered by OpenCV & Streamlit
</p>
""", unsafe_allow_html=True)
