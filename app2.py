from src.utils import read_config
from src.segmentation import digits_segmentaion
from src.ocr import ocr
import cv2
import streamlit as st

config = read_config()

def app():
    st.title("OCR App")
    camera_placeholder = st.empty()

    # Initialize variables
    camera_focus = False
    first_shot = True
    drawing = False
    start_x, start_y = -1, -1
    end_x, end_y = -1, -1
    frame = None

    # Mouse callback function
    def draw_roi(event, x, y, flags, param):
        nonlocal frame, start_x, start_y, drawing, end_x, end_y
        img = frame.copy()
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            start_x, start_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(img, (start_x, start_y), (x, y), (0, 255, 0), 2)
            camera_placeholder.image(img, channels="BGR")
            start_x = start_x + 3
            start_y = start_y + 3
            end_x = x - 3
            end_y = y - 3
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            img_copy = img.copy()
            cv2.rectangle(img_copy, (start_x, start_y), (x, y), (0, 255, 0), 2)
            camera_placeholder.image(img_copy, channels="BGR")

    # Using Phone Camera
    cap = cv2.VideoCapture('http://192.168.0.35:4747/video')

    if not cap.isOpened():
        st.error("Cannot open camera!")
        return

    cv2.namedWindow('Camera')
    cv2.setMouseCallback('Camera', draw_roi)

    while True:
        ret, frame = cap.read()
        if ret:
            if not camera_focus:  # Setting up the camera focus
                cv2.putText(frame, "Focus Your Camera", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
                cv2.putText(frame, "(Push 'space' when focus is adjusted)", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                camera_placeholder.image(frame, channels="BGR")
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    st.write("Focus is adjusted")
                    camera_focus = True
            else:
                if first_shot:  # Drawing RoI
                    cv2.putText(frame, "Draw RoI", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
                    cv2.putText(frame, "(Double push 'space' when RoI is drawn)", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    camera_placeholder.image(frame, channels="BGR")
                    cv2.waitKey(0)
                    first_shot = False
                    st.write("Press space when RoI is finished")

            # Check for key press or mouse event
            key = cv2.waitKey(1) & 0xFF

            # Perform OCR if space key is pressed or right mouse button is clicked
            if key == ord(' ') or key == cv2.EVENT_RBUTTONDOWN:
                roi_img = frame[start_y:end_y, start_x:end_x]
                try:
                    digits_concat, contours_min_location = digits_segmentaion(roi_img,
                                                        config["inference"]["segmentation_threshold_value"],
                                                        config["inference"]["segmentation_digit_min_area"],
                                                        config["inference"]["segmentation_digit_crop_offset"])
                    labels = ocr(digits_concat)
                    st.write(f"Recognized digits: {labels}")
                    st.write(f"Contour locations: {contours_min_location}")
                except Exception as e:
                    st.write(f"Error: {e}")
                    st.write("Digit not found")
                # Draw the recognized digits on the frame
                l = 0
                for loc in contours_min_location:
                    cv2.putText(frame, str(labels[l]), (loc[0]+start_x, loc[1]+start_y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    l += 1
                camera_placeholder.image(frame, channels="BGR")

            # Exit the loop if 'q' is pressed
            if key == ord('q'):
                break
        else:
            st.error("Camera feed stopped.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    app()