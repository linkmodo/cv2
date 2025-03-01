# Define a constant for no rotation
ROTATE_NONE = None

# For image processing:
rotation_angle = st.selectbox(
    "Rotate Image",
    [ROTATE_NONE, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180],
    format_func=lambda x: "None" if x is None else (
        "90° CW" if x == cv2.ROTATE_90_CLOCKWISE else 
        "90° CCW" if x == cv2.ROTATE_90_COUNTERCLOCKWISE else 
        "180°"
    )
)

def rotate_image(image, angle):
    if angle == cv2.ROTATE_90_CLOCKWISE:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == cv2.ROTATE_180:
        return cv2.rotate(image, cv2.ROTATE_180)
    else:
        return image  # No rotation

# For video processing, use the same approach:
rotation_angle = st.selectbox(
    "Rotate Video",
    [ROTATE_NONE, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180],
    format_func=lambda x: "None" if x is None else (
        "90° CW" if x == cv2.ROTATE_90_CLOCKWISE else 
        "90° CCW" if x == cv2.ROTATE_90_COUNTERCLOCKWISE else 
        "180°"
    )
)
