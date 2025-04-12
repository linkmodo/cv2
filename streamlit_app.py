import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import tempfile
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import queue
import threading

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

# Initialize session states
if 'rotation_angle_image' not in st.session_state:
    st.session_state.rotation_angle_image = None
if 'rotation_angle_video' not in st.session_state:
    st.session_state.rotation_angle_video = None

# ---------------------
# Main Content Area with Tabs
# ---------------------
detection_tab = st.tabs(["Image", "Video", "Webcam"])

# ---------------------
# Image Detection Tab
# ---------------------
with detection_tab[0]:
    st.header("Image Face Detection")
    
    # Sidebar settings for Image
    with st.sidebar:
        st.header("Image Detection Settings")
        conf_threshold_img = st.slider("Confidence Threshold (Image)", 0.0, 1.0, 0.5, 0.01)
        box_color_hex = st.color_picker("Bounding Box Color (Image)", "#00FF00")
        thickness_img = st.slider("Bounding Box Thickness (Image)", 1, 10, 4)
        # Convert hex to BGR tuple
        box_color_img = tuple(int(box_color_hex[i:i+2], 16) for i in (1, 3, 5))
        box_color_img = (box_color_img[2], box_color_img[1], box_color_img[0])
        
        # Image rotation controls
        st.subheader("Rotation Options")
        r1, r2, r3, r4 = st.columns(4)
        if r1.button("Rotate 90° CW (Image)"):
            st.session_state.rotation_angle_image = cv2.ROTATE_90_CLOCKWISE
        if r2.button("Rotate 90° CCW (Image)"):
            st.session_state.rotation_angle_image = cv2.ROTATE_90_COUNTERCLOCKWISE
        if r3.button("Rotate 180° (Image)"):
            st.session_state.rotation_angle_image = cv2.ROTATE_180
        if r4.button("Reset (Image)"):
            st.session_state.rotation_angle_image = None
    
    # Image upload and processing
    img_file_buffer = st.file_uploader("Upload an image file with face(s) in it to be analyzed", type=['jpg', 'jpeg', 'png'])
    if img_file_buffer is not None:
        raw_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
        image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        
        col1, col2 = st.columns(2)
        col1.image(image, channels='BGR', caption="Input Image")
        
        rotated_image = rotate_image(image, st.session_state.rotation_angle_image)
        detections = detectFaceOpenCVDnn(net, rotated_image)
        out_image = process_detections(rotated_image.copy(), detections, conf_threshold_img, box_color_img, thickness_img)
        
        col2.image(out_image, channels='BGR', caption="Output Image")
        out_image_pil = Image.fromarray(cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        out_image_pil.save(buf, format='JPEG')
        st.download_button("Download Processed Image", data=buf.getvalue(), file_name="processed_image.jpg", mime="image/jpeg")

# ---------------------
# Video Detection Tab
# ---------------------
with detection_tab[1]:
    st.header("Video Face Detection")
    
    # Sidebar settings for Video
    with st.sidebar:
        st.header("Video Detection Settings")
        conf_threshold_video = st.slider("Confidence Threshold (Video)", 0.0, 1.0, 0.5, 0.01)
        box_color_video_hex = st.color_picker("Bounding Box Color (Video)", "#00FF00")
        thickness_video = st.slider("Bounding Box Thickness (Video)", 1, 10, 4)
        box_color_video = tuple(int(box_color_video_hex[i:i+2], 16) for i in (1, 3, 5))
        box_color_video = (box_color_video[2], box_color_video[1], box_color_video[0])
        
        # Video rotation controls
        st.subheader("Rotation Options")
        vr1, vr2, vr3, vr4 = st.columns(4)
        if vr1.button("Rotate 90° CW (Video)"):
            st.session_state.rotation_angle_video = cv2.ROTATE_90_CLOCKWISE
        if vr2.button("Rotate 90° CCW (Video)"):
            st.session_state.rotation_angle_video = cv2.ROTATE_90_COUNTERCLOCKWISE
        if vr3.button("Rotate 180° (Video)"):
            st.session_state.rotation_angle_video = cv2.ROTATE_180
        if vr4.button("Reset (Video)"):
            st.session_state.rotation_angle_video = None
    
    # Video upload and processing
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
# Webcam Detection Tab
# ---------------------
with detection_tab[2]:
    st.header("Webcam Face Detection")
    
    # Sidebar settings for Webcam
    with st.sidebar:
        st.header("Webcam Detection Settings")
        conf_threshold_webcam = st.slider("Confidence Threshold (Webcam)", 0.0, 1.0, 0.5, 0.01)
        box_color_webcam_hex = st.color_picker("Bounding Box Color (Webcam)", "#00FF00")
        thickness_webcam = st.slider("Bounding Box Thickness (Webcam)", 1, 10, 4)
        # Convert hex to BGR tuple
        box_color_webcam = tuple(int(box_color_webcam_hex[i:i+2], 16) for i in (1, 3, 5))
        box_color_webcam = (box_color_webcam[2], box_color_webcam[1], box_color_webcam[0])
        show_confidence = st.checkbox("Show Confidence Score", True)
    
    # Define RTC configuration with free STUN servers
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Define a threadsafe class for webcam processing
    class VideoProcessor(VideoProcessorBase):
        result_queue = queue.Queue()
        
        def __init__(self) -> None:
            self.confidence_threshold = conf_threshold_webcam
            self.box_color = box_color_webcam
            self.thickness = thickness_webcam
            self.show_confidence = show_confidence
            self._model = net
            self.frame_lock = threading.Lock()
            self.in_progress = False
        
        def _detect_faces(self, frame: av.VideoFrame) -> np.ndarray:
            img = frame.to_ndarray(format="bgr24")
            
            # Face detection
            blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
            self._model.setInput(blob)
            detections = self._model.forward()
            
            # Draw bounding boxes
            img_h, img_w = img.shape[:2]
            detected_faces = []
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > self.confidence_threshold:
                    x1 = int(detections[0, 0, i, 3] * img_w)
                    y1 = int(detections[0, 0, i, 4] * img_h)
                    x2 = int(detections[0, 0, i, 5] * img_w)
                    y2 = int(detections[0, 0, i, 6] * img_h)
                    
                    # Ensure coordinates are within frame boundaries
                    x1 = max(0, min(x1, img_w - 1))
                    y1 = max(0, min(y1, img_h - 1))
                    x2 = max(0, min(x2, img_w - 1))
                    y2 = max(0, min(y2, img_h - 1))
                    
                    # Store detected face
                    detected_faces.append((x1, y1, x2, y2, confidence))
                    
                    # Draw rectangle
                    cv2.rectangle(img, (x1, y1), (x2, y2), self.box_color, self.thickness, cv2.LINE_8)
                    
                    # Show confidence if enabled
                    if self.show_confidence:
                        label = f"{confidence:.2f}"
                        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        cv2.rectangle(img, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), self.box_color, cv2.FILLED)
                        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Put detection results in queue for display in Streamlit
            if len(detected_faces) > 0:
                self.result_queue.put(len(detected_faces))
            
            return img
        
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            with self.frame_lock:
                if self.in_progress:
                    return frame
                self.in_progress = True
            
            # Update params from Streamlit's sidebar
            self.confidence_threshold = conf_threshold_webcam
            self.box_color = box_color_webcam
            self.thickness = thickness_webcam
            self.show_confidence = show_confidence
            
            # Process the frame
            img = self._detect_faces(frame)
            
            with self.frame_lock:
                self.in_progress = False
            
            # Return the processed image
            return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    # Create the webrtc streamer component
    ctx = webrtc_streamer(
        key="face-detection-webcam",
        video_processor_factory=VideoProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    if ctx.video_processor:
        # Display real-time detection statistics
        if not ctx.video_processor.result_queue.empty():
            result = ctx.video_processor.result_queue.get()
            st.session_state.face_count = result
        
        face_count = st.session_state.get("face_count", 0)
        if face_count > 0:
            st.success(f"Number of faces detected: {face_count}")
    
    # User instructions
    st.info("👆 Click START above to activate your webcam. Make sure to allow camera access when prompted.")
    st.warning("If you don't see detection boxes, try adjusting the settings in the sidebar.")

# ---------------------
# Footer
# ---------------------
st.markdown("""
<hr>
<p style="text-align: center; color: gray;">
Built by Li Fan 2025-03-01 | Powered by OpenCV & Streamlit
</p>
""", unsafe_allow_html=True)
