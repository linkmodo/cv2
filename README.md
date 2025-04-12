# Deep Learning Based Face Detection and Privacy Filter

A Streamlit application that uses OpenCV's deep learning model to detect faces in images, videos, and webcam streams, with privacy filtering options.

![Face Detection Demo](https://raw.githubusercontent.com/linkmodo/cv2/main/theatre-4981934.jpg)

## Features

- **Image Processing**: Upload an image and detect faces with customizable confidence threshold and bounding box styles
- **Video Processing**: Upload a video file for face detection with adjustable parameters
- **Webcam Integration**: Real-time face detection through your webcam with high-performance streaming
- **Privacy Filtering**: Automatically apply mosaic blur to detected faces in images, videos, and webcam streams
- **Improved UI**: Intuitive tabbed interface with collapsible settings panels for a cleaner user experience
- **Advanced Customization**: 
  - Adjust confidence thresholds for detection accuracy
  - Customize bounding box colors and thicknesses
  - Control privacy filter strength with adjustable mosaic block size
- **Rotation Controls**: Rotate images and videos with 90° and 180° options
- **Download Capability**: Save processed images and videos with privacy filters applied

## Recent Improvements

### UI/UX Enhancements
- **Tabbed Interface**: Separated image, video, and webcam functionality into dedicated tabs
- **Collapsible Settings**: Organized settings into expandable/collapsible panels
- **Optimized Layout**: Two-column design for more efficient use of screen space
- **Consistent Controls**: Standardized parameter controls across all three detection modes

### Webcam Streaming Improvements
- **Robust WebRTC Implementation**: Fixed stability issues with webcam streaming
- **Optimized Performance**: Better frame processing for smoother real-time detection
- **Thread-Safe Processing**: Improved handling of video frames for more reliable detection
- **Real-Time Statistics**: Displays the number of faces detected in the webcam stream

### Privacy Features
- **Mosaic Privacy Filter**: Apply pixelation to detected faces to preserve privacy
- **Adjustable Privacy Level**: Control the intensity of the privacy filter
- **Real-Time Application**: Privacy filtering works in real-time for webcam streams
- **Consistent Experience**: Same privacy options available for images, videos, and webcam

## Requirements

- Python 3.7+
- Streamlit
- OpenCV
- NumPy
- Pillow
- streamlit-webrtc
- av

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/face-detection-app.git
   cd face-detection-app
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

## Model Details

This application uses the OpenCV DNN face detector model:
- Model file: `res10_300x300_ssd_iter_140000_fp16.caffemodel`
- Configuration: `deploy.prototxt`

The model is a Single Shot MultiBox Detector (SSD) with a ResNet base network trained for face detection.

## Usage

1. **Image Detection**:
   - Select the Image tab
   - Upload an image using the file uploader
   - Expand the settings panel to adjust confidence threshold and bounding box settings
   - Toggle the privacy filter to anonymize faces
   - Apply rotation if needed
   - Download the processed image

2. **Video Detection**:
   - Select the Video tab
   - Upload a video file
   - Expand the settings panel to adjust detection and privacy parameters
   - View processing progress
   - Download the processed video

3. **Webcam Detection**:
   - Select the Webcam tab
   - Expand the settings panel to customize detection and privacy options
   - Click the "START" button to activate your webcam
   - Real-time face detection and privacy filtering will be displayed

## Privacy

This application processes all data locally and does not save or transmit any user data. The privacy filter feature adds an additional layer of protection by allowing users to anonymize faces in their media.

## License

[MIT License](LICENSE)

## Acknowledgements

- OpenCV for the pre-trained face detection model
- Streamlit for the web application framework
- streamlit-webrtc for reliable webcam streaming 
