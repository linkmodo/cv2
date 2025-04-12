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
st.set_page_config(page_title="Deep Learning Face Detection Model using OpenCV", layout="wide")
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
    background: -webkit-linear-gradient(75deg, orange, yellow);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    color: black;">
Deep Learning Based Face Detection
</h1>
""", unsafe_allow_html=True)

st.markdown("""
<h3 style="text-align: center; color: white; font-size: 20px;">
This app detects face(s) in images and videos using OpenCV's deep learning model.<br>
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
# Video Processing Section
# ---------------------
video_file_buffer = st.file_uploader("Upload a video file with face(s) in it to be analyzed", type=['mp4', 'avi', 'mov'])
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
    
    conf_threshold_video = st.slider("Confidence Threshold (Video)", 0.0, 1.0, 0.5, 0.01)
    box_color_video_hex = st.color_picker("Bounding Box Color (Video)", "#00FF00")
    thickness_video = st.slider("Bounding Box Thickness (Video)", 1, 10, 4)
    box_color_video = tuple(int(box_color_video_hex[i:i+2], 16) for i in (1, 3, 5))
    box_color_video = (box_color_video[2], box_color_video[1], box_color_video[0])
    
    if 'rotation_angle_video' not in st.session_state:
        st.session_state.rotation_angle_video = None

    vr1, vr2, vr3, vr4 = st.columns(4)
    if vr1.button("Rotate 90° CW (Video)"):
        st.session_state.rotation_angle_video = cv2.ROTATE_90_CLOCKWISE
    if vr2.button("Rotate 90° CCW (Video)"):
        st.session_state.rotation_angle_video = cv2.ROTATE_90_COUNTERCLOCKWISE
    if vr3.button("Rotate 180° (Video)"):
        st.session_state.rotation_angle_video = cv2.ROTATE_180
    if vr4.button("Reset Rotation (Video)"):
        st.session_state.rotation_angle_video = None

    rotation_angle_video = st.session_state.rotation_angle_video
    out_width = width
    out_height = height
    if rotation_angle_video in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
        out_width, out_height = height, width
    
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
        processed_frame = process_detections(frame.copy(), detections, conf_threshold_video, box_color_video, thickness_video)
        out.write(processed_frame)
        stframe.image(processed_frame, channels="BGR")
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
# Webcam Face Detection
# ---------------------
st.markdown("<h2 style='text-align: center; color: white;'>Webcam Face Detection</h2>", unsafe_allow_html=True)

# Webcam settings
conf_threshold_webcam = st.slider("Confidence Threshold (Webcam)", 0.0, 1.0, 0.5, 0.01)
box_color_webcam_hex = st.color_picker("Bounding Box Color (Webcam)", "#00FF00")
thickness_webcam = st.slider("Bounding Box Thickness (Webcam)", 1, 10, 4)
# Convert hex to BGR tuple
box_color_webcam = tuple(int(box_color_webcam_hex[i:i+2], 16) for i in (1, 3, 5))
box_color_webcam = (box_color_webcam[2], box_color_webcam[1], box_color_webcam[0])

# Import streamlit-webrtc for better webcam handling
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av

# Define RTC configuration with free STUN servers
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Define video transformer for face detection
class FaceDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Access current widget values inside the method
        threshold = st.session_state.get("conf_threshold_webcam", 0.5)
        box_color = st.session_state.get("box_color_webcam", (0, 255, 0))
        thickness = st.session_state.get("thickness_webcam", 2)

        # Create blob and run detection
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()
        
        # Draw bounding boxes
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
                label = f"{confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                y1 = max(y1, label_size[1])
                cv2.rectangle(img, (x1, y1 - label_size[1]), (x1 + label_size[0], y1), box_color, cv2.FILLED)
                cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Create the webrtc streamer component
webrtc_ctx = webrtc_streamer(
    key="face-detection",
    video_transformer_factory=FaceDetectionTransformer,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Update the parameters when they change
if webrtc_ctx.video_transformer:
    webrtc_ctx.video_transformer.threshold = conf_threshold_webcam
    webrtc_ctx.video_transformer.box_color = box_color_webcam
    webrtc_ctx.video_transformer.thickness = thickness_webcam

# Instructions for users
st.info("If the webcam doesn't start automatically, check your browser permissions and try clicking the 'START' button. Allow camera access when prompted.")

# ---------------------
# Footer
# ---------------------
st.markdown("""
<hr>
<p style="text-align: center; color: gray;">
Built by Li Fan 2025-03-01 | Powered by OpenCV & Streamlit
</p>
""", unsafe_allow_html=True)
