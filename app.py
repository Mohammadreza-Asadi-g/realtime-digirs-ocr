import cv2
import streamlit as st
from inference import inference_ocr

# Function to get the camera frames
def get_camera_frames():
    camera = cv2.VideoCapture('http://192.168.0.35:4747/video')  # 0 for the default camera
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield frame

# Streamlit app
def app():
    st.title("Camera Feed")
    camera_frames = get_camera_frames()
    frame_placeholder = st.empty()  # Create a placeholder for the frame

    while True:
        try:
            frame = next(camera_frames)
            frame_placeholder.image(frame, channels="BGR")  # Update the placeholder with the new frame
        except StopIteration:
            st.error("Camera feed stopped.")
            break

if __name__ == "__main__":
    app()