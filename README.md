# Deep Learning Based Face Detection

A Streamlit application that uses OpenCV's deep learning model to detect faces in images, videos, and webcam streams.

![Face Detection Demo](https://raw.githubusercontent.com/linkmodo/cv2/main/theatre-4981934.jpg)

## Features

- **Image Processing**: Upload an image and detect faces with customizable confidence threshold and bounding box styles
- **Video Processing**: Upload a video file for face detection with adjustable parameters
- **Webcam Integration**: Real-time face detection through your webcam
- **Customization Options**: Adjust confidence thresholds, bounding box colors, and thicknesses
- **Rotation Controls**: Rotate images and videos with 90° and 180° options
- **Download Capability**: Save processed images and videos

## Requirements

- Python 3.7+
- Streamlit
- OpenCV
- NumPy
- Pillow

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
   - Upload an image using the file uploader
   - Adjust confidence threshold and bounding box settings
   - Apply rotation if needed
   - Download the processed image

2. **Video Detection**:
   - Upload a video file
   - Adjust detection parameters
   - View processing progress
   - Download the processed video

3. **Webcam Detection**:
   - Click "Start/Stop Webcam" button
   - Adjust detection parameters
   - Real-time face detection will be displayed

## Privacy

This application processes all data locally and does not save or transmit any user data after exiting.

## License

[MIT License](LICENSE)

## Acknowledgements

- OpenCV for the pre-trained face detection model
- Streamlit for the web application framework 
