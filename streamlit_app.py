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

def process_detections(frame, detections, conf_threshold=0.5, box_color=(0, 255, 0), thickness=2, apply_mosaic=False, mosaic_level=10):
    frame_h, frame_w = frame.shape[:2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_w)
            y1 = int(detections[0, 0, i, 4] * frame_h)
            x2 = int(detections[0, 0, i, 5] * frame_w)
            y2 = int(detections[0, 0, i, 6] * frame_h)
            
            # Ensure coordinates are within frame boundaries
            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(0, min(x2, frame_w - 1))
            y2 = max(0, min(y2, frame_h - 1))
            
            # Apply mosaic blur if requested
            if apply_mosaic and x2 > x1 and y2 > y1:
                face_roi = frame[y1:y2, x1:x2].copy()
                # Apply pixelation effect (mosaic blur)
                h, w = face_roi.shape[:2]
                # Reduce size to create pixelation effect
                temp = cv2.resize(face_roi, (max(1, w // mosaic_level), max(1, h // mosaic_level)), 
                                 interpolation=cv2.INTER_LINEAR)
                # Resize back to original size with nearest neighbor to maintain blocks
                pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
                frame[y1:y2, x1:x2] = pixelated
                
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
    
    # Settings in collapsible expander
    with st.expander("Image Detection Settings", expanded=False, key="img_settings_expander"):
        col1, col2 = st.columns(2)
        
        with col1:
            conf_threshold_img = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01, key="conf_img")
            box_color_hex = st.color_picker("Bounding Box Color", "#00FF00", key="color_img")
            thickness_img = st.slider("Bounding Box Thickness", 1, 10, 4, key="thick_img")
        
        with col2:
            # Add mosaic privacy filter option
            apply_mosaic_img = st.checkbox("Apply Mosaic Filter", False, key="mosaic_img_check")
            if apply_mosaic_img:
                mosaic_level_img = st.slider("Mosaic Block Size", 5, 50, 15, 1, key="mosaic_img_level")
                
            # Image rotation controls in sub-columns
            st.subheader("Rotation Options")
            rot1, rot2 = st.columns(2)
            if rot1.button("Rotate 90° CW", key="rot90cw_img"):
                st.session_state.rotation_angle_image = cv2.ROTATE_90_CLOCKWISE
            if rot2.button("Rotate 90° CCW", key="rot90ccw_img"):
                st.session_state.rotation_angle_image = cv2.ROTATE_90_COUNTERCLOCKWISE
            
            rot3, rot4 = st.columns(2)
            if rot3.button("Rotate 180°", key="rot180_img"):
                st.session_state.rotation_angle_image = cv2.ROTATE_180
            if rot4.button("Reset Rotation", key="rotreset_img"):
                st.session_state.rotation_angle_image = None
    
    # Convert hex to BGR tuple
    box_color_img = tuple(int(box_color_hex[i:i+2], 16) for i in (1, 3, 5))
    box_color_img = (box_color_img[2], box_color_img[1], box_color_img[0])
    
    # Image upload and processing
    img_file_buffer = st.file_uploader("Upload an image file with face(s) in it to be analyzed", 
                                      type=['jpg', 'jpeg', 'png'], 
                                      key="img_uploader")
    if img_file_buffer is not None:
        raw_bytes = np.asarray(bytearray(img_file_buffer.read()), dtype=np.uint8)
        image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        
        display_col1, display_col2 = st.columns(2)
        display_col1.image(image, channels='BGR', caption="Input Image")
        
        rotated_image = rotate_image(image, st.session_state.rotation_angle_image)
        detections = detectFaceOpenCVDnn(net, rotated_image)
        out_image = process_detections(
            rotated_image.copy(), 
            detections, 
            conf_threshold_img, 
            box_color_img, 
            thickness_img,
            apply_mosaic_img,
            mosaic_level_img if apply_mosaic_img else 10
        )
        
        display_col2.image(out_image, channels='BGR', caption="Output Image")
        out_image_pil = Image.fromarray(cv2.cvtColor(out_image, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        out_image_pil.save(buf, format='JPEG')
        st.download_button("Download Processed Image", 
                          data=buf.getvalue(), 
                          file_name="processed_image.jpg", 
                          mime="image/jpeg",
                          key="img_download")

# ---------------------
# Video Detection Tab
# ---------------------
with detection_tab[1]:
    st.header("Video Face Detection")
    
    # Settings in collapsible expander
    with st.expander("Video Detection Settings", expanded=False, key="video_settings_expander"):
        col1, col2 = st.columns(2)
        
        with col1:
            conf_threshold_video = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01, key="conf_video")
            box_color_video_hex = st.color_picker("Bounding Box Color", "#00FF00", key="color_video")
            thickness_video = st.slider("Bounding Box Thickness", 1, 10, 4, key="thick_video")
        
        with col2:
            # Add mosaic privacy filter option
            apply_mosaic_video = st.checkbox("Apply Mosaic Filter", False, key="mosaic_video_check")
            if apply_mosaic_video:
                mosaic_level_video = st.slider("Mosaic Block Size", 5, 50, 15, 1, key="mosaic_video_level")
            
            # Video rotation controls in sub-columns
            st.subheader("Rotation Options")
            vrot1, vrot2 = st.columns(2)
            if vrot1.button("Rotate 90° CW", key="rot90cw_video"):
                st.session_state.rotation_angle_video = cv2.ROTATE_90_CLOCKWISE
            if vrot2.button("Rotate 90° CCW", key="rot90ccw_video"):
                st.session_state.rotation_angle_video = cv2.ROTATE_90_COUNTERCLOCKWISE
            
            vrot3, vrot4 = st.columns(2)
            if vrot3.button("Rotate 180°", key="rot180_video"):
                st.session_state.rotation_angle_video = cv2.ROTATE_180
            if vrot4.button("Reset Rotation", key="rotreset_video"):
                st.session_state.rotation_angle_video = None
    
    # Convert hex to BGR tuple
    box_color_video = tuple(int(box_color_video_hex[i:i+2], 16) for i in (1, 3, 5))
    box_color_video = (box_color_video[2], box_color_video[1], box_color_video[0])
    
    # Video upload and processing
    video_file_buffer = st.file_uploader("Upload a video file with face(s) in it to be analyzed", 
                                        type=['mp4', 'avi', 'mov'], 
                                        key="video_uploader")
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
        progress_bar = st.progress(0, key="video_progress")
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
            progress_bar.progress(progress, key="video_progress_update")
            frame = rotate_image(frame, rotation_angle_video)
            detections = detectFaceOpenCVDnn(net, frame)
            processed_frame = process_detections(
                frame.copy(), 
                detections, 
                conf_threshold_video, 
                box_color_video, 
                thickness_video,
                apply_mosaic_video,
                mosaic_level_video if apply_mosaic_video else 10
            )
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
            st.download_button("Download Processed Video", 
                              f, 
                              file_name="processed_video.mp4", 
                              mime="video/mp4",
                              key="video_download")
        try:
            os.unlink(output_path)
        except Exception as e:
            st.error(f"Error deleting temporary file: {e}")

# ---------------------
# Webcam Detection Tab
# ---------------------
with detection_tab[2]:
    st.header("Webcam Face Detection")
    
    # Settings in collapsible expander
    with st.expander("Webcam Detection Settings", expanded=False, key="webcam_settings_expander"):
        col1, col2 = st.columns(2)
        
        with col1:
            conf_threshold_webcam = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01, key="conf_webcam")
            box_color_webcam_hex = st.color_picker("Bounding Box Color", "#00FF00", key="color_webcam")
            thickness_webcam = st.slider("Bounding Box Thickness", 1, 10, 4, key="thick_webcam")
            show_confidence = st.checkbox("Show Confidence Score", True, key="show_conf_webcam")
        
        with col2:
            # Add mosaic privacy filter option
            apply_mosaic_webcam = st.checkbox("Apply Mosaic Filter", False, key="mosaic_webcam_check")
            if apply_mosaic_webcam:
                mosaic_level_webcam = st.slider("Mosaic Block Size", 5, 50, 15, 1, key="mosaic_webcam_level")
    
    # Convert hex to BGR tuple
    box_color_webcam = tuple(int(box_color_webcam_hex[i:i+2], 16) for i in (1, 3, 5))
    box_color_webcam = (box_color_webcam[2], box_color_webcam[1], box_color_webcam[0])
    
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
            # Privacy filter
            self.apply_mosaic = apply_mosaic_webcam
            self.mosaic_level = mosaic_level_webcam if apply_mosaic_webcam else 10
        
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
                    
                    # Apply mosaic if requested
                    if self.apply_mosaic and x2 > x1 and y2 > y1:
                        face_roi = img[y1:y2, x1:x2].copy()
                        # Apply pixelation effect (mosaic blur)
                        h, w = face_roi.shape[:2]
                        # Reduce size to create pixelation effect
                        temp = cv2.resize(face_roi, (max(1, w // self.mosaic_level), max(1, h // self.mosaic_level)), 
                                         interpolation=cv2.INTER_LINEAR)
                        # Resize back to original size with nearest neighbor to maintain blocks
                        pixelated = cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
                        img[y1:y2, x1:x2] = pixelated
                    
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
            # Update privacy filter settings
            self.apply_mosaic = apply_mosaic_webcam
            self.mosaic_level = mosaic_level_webcam if apply_mosaic_webcam else 10
            
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
            st.success(f"Number of faces detected: {face_count}", key="webcam_face_count")
    
    # User instructions
    st.info("👆 Click START above to activate your webcam. Make sure to allow camera access when prompted.", key="webcam_info")
    st.warning("If you don't see detection boxes, try adjusting the settings in the expandable section above.", key="webcam_warning")

# ---------------------
# Footer
# ---------------------
st.markdown("""
<hr>
<p style="text-align: center; color: gray;">
Built by Li Fan 2025-03-01 | Powered by OpenCV & Streamlit
</p>
""", unsafe_allow_html=True)
