from src.utils import read_config
from src.segmentation import digits_segmentaion
from src.ocr import ocr
import cv2 as cv

config = read_config()

# Initialize variables
frame_crop = False
camera_focus = False
first_shot = True
drawing = False
start_x, start_y = -1, -1
end_x, end_y = -1, -1

# Mouse callback function
def draw_roi(event, x, y, flags, param):
    global frame, start_x, start_y, drawing, end_x, end_y
    img = frame
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        cv.rectangle(img, (start_x, start_y), (x, y), (0, 255, 0), 2)
        cv.imshow('Camera', img)
        cv.imwrite("r.png", img[start_y+3:y-3, start_x+3:x-3])
        start_x = start_x+3; start_y = start_y+3; end_x = x-3; end_y = y-3
    elif event == cv.EVENT_MOUSEMOVE and drawing:
        img_copy = img.copy()
        cv.rectangle(img_copy, (start_x, start_y), (x, y), (0, 255, 0), 2)
        cv.imshow('Camera', img_copy)

# Using Phone Camera
cap = cv.VideoCapture('http://192.168.0.35:4747/video')

if not cap.isOpened():
    print("Cannot open camera!")
    exit()

# Set the mouse callback function
cv.namedWindow('Camera')
cv.setMouseCallback('Camera', draw_roi)

while True:
    ret, frame = cap.read()
    if ret:
        if camera_focus == False:  # Setting up the camera focus
            cv.putText(frame, "Focus Your Camera", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
            cv.putText(frame, "(Push 'space' when focus is adjusted)", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv.imshow('Camera', frame)
            if cv.waitKey(1) & 0xFF == ord(' '):
                print("Focus is adjusted")
                camera_focus = True
        else:
            if first_shot:  # Drawing RoI
                cv.putText(frame, "Draw Table RoI", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
                cv.putText(frame, "(Push 'space' when RoI is drawn)", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv.imshow('Camera', frame)
                cv.waitKey(0)
                first_shot = False
                print("Press space when RoI is finished")

        # Check for key press or mouse event
        key = cv.waitKey(1) & 0xFF

        # Perform OCR if space key is pressed or right mouse button is clicked
        if key == ord(' ') or key == cv.EVENT_RBUTTONDOWN:
            cv.imshow('Camera', frame)
            roi_img = frame[start_y:end_y, start_x:end_x]
            digits_concat = digits_segmentaion(roi_img,
                                               config["inference"]["segmentation_threshold_value"],
                                               config["inference"]["segmentation_digit_min_area"],
                                               config["inference"]["segmentation_digit_crop_offset"])
            labels = ocr(digits_concat)
            print(labels)

            # Draw the recognized digits on the frame
            cv.putText(frame, str(labels), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv.imshow('Camera', frame)


        # Exit the loop if 'q' is pressed
        if key == ord('q'):
            break
    else:
        break

cap.release()
cv.destroyAllWindows()