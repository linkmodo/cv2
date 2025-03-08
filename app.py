import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import tempfile

def apply_box_blur(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))

def apply_gaussian_blur(image, kernel_size, sigma=0):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def apply_sharpen(image, intensity='normal'):
    if intensity == 'normal':
        kernel = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])
    else:  # intense
        kernel = np.array([[0, -4, 0],
                          [-4, 17, -4],
                          [0, -4, 0]])
    return cv2.filter2D(image, -1, kernel)

def process_image(image, operation, params):
    if operation == "Box Blur":
        return apply_box_blur(image, params["kernel_size"])
    elif operation == "Gaussian Blur":
        return apply_gaussian_blur(image, params["kernel_size"], params.get("sigma", 0))
    elif operation == "Sharpen":
        return apply_sharpen(image, params.get("intensity", "normal"))
    return image

def main():
    st.title("Image Processing Application")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "jpeg", "png", "mp4"])
    
    if uploaded_file is not None:
        # Determine if it's an image or video
        file_type = uploaded_file.type
        
        if file_type.startswith('image'):
            # Handle image processing
            image = Image.open(uploaded_file)
            image = np.array(image)
            
            # Convert BGR to RGB for display
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            st.image(image, caption="Original Image", channels="BGR")
            
            # Sidebar controls
            st.sidebar.header("Processing Options")
            operation = st.sidebar.selectbox(
                "Select Operation",
                ["Box Blur", "Gaussian Blur", "Sharpen"]
            )
            
            params = {}
            if operation in ["Box Blur", "Gaussian Blur"]:
                params["kernel_size"] = st.sidebar.slider(
                    "Kernel Size",
                    3, 31, 5, step=2
                )
                if operation == "Gaussian Blur":
                    params["sigma"] = st.sidebar.slider(
                        "Sigma",
                        0, 10, 0
                    )
            elif operation == "Sharpen":
                params["intensity"] = st.sidebar.selectbox(
                    "Sharpening Intensity",
                    ["normal", "intense"]
                )
            
            if st.sidebar.button("Apply Filter"):
                processed_image = process_image(image, operation, params)
                st.image(processed_image, caption="Processed Image", channels="BGR")
                
                # Create download button for processed image
                processed_pil = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
                buf = io.BytesIO()
                processed_pil.save(buf, format="PNG")
                st.download_button(
                    label="Download Processed Image",
                    data=buf.getvalue(),
                    file_name="processed_image.png",
                    mime="image/png"
                )
                
        elif file_type.startswith('video'):
            # Handle video processing
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            st.sidebar.header("Processing Options")
            operation = st.sidebar.selectbox(
                "Select Operation",
                ["Box Blur", "Gaussian Blur", "Sharpen"]
            )
            
            params = {}
            if operation in ["Box Blur", "Gaussian Blur"]:
                params["kernel_size"] = st.sidebar.slider(
                    "Kernel Size",
                    3, 31, 5, step=2
                )
                if operation == "Gaussian Blur":
                    params["sigma"] = st.sidebar.slider(
                        "Sigma",
                        0, 10, 0
                    )
            elif operation == "Sharpen":
                params["intensity"] = st.sidebar.selectbox(
                    "Sharpening Intensity",
                    ["normal", "intense"]
                )
            
            if st.sidebar.button("Process Video"):
                video = cv2.VideoCapture(tfile.name)
                
                # Get video properties
                width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(video.get(cv2.CAP_PROP_FPS))
                
                # Create temporary file for processed video
                processed_video_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                out = cv2.VideoWriter(
                    processed_video_file.name,
                    cv2.VideoWriter_fourcc(*'mp4v'),
                    fps, (width, height)
                )
                
                # Process video
                progress_bar = st.progress(0)
                frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                
                for i in range(frame_count):
                    ret, frame = video.read()
                    if ret:
                        processed_frame = process_image(frame, operation, params)
                        out.write(processed_frame)
                        progress_bar.progress((i + 1) / frame_count)
                
                video.release()
                out.release()
                
                # Provide download link for processed video
                with open(processed_video_file.name, 'rb') as f:
                    st.download_button(
                        label="Download Processed Video",
                        data=f.read(),
                        file_name="processed_video.mp4",
                        mime="video/mp4"
                    )

if __name__ == "__main__":
    main() 
