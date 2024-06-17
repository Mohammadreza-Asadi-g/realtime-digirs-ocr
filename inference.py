from src.utils import read_config
from src.segmentation import digits_segmentaion
from src.ocr import ocr
import cv2 as cv
import numpy as np


config = read_config()


# Initialize variables
drawing = False
roi = None

# Mouse callback function
def draw_roi(event, x, y, flags, param):
    global drawing, roi, roi_points

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        roi_points = [(x, y)]

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            roi_points.append((x, y))

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        roi_points.append((x, y))
        roi = [roi_points]


# Using Phone Camera
cap  = cv.VideoCapture('http://192.168.0.35:4747/video') 

if not cap.isOpened():
    print("Cannot open camera!")
    exit()

# Flag to indicate if OCR should be performed
# perform_ocr_flag = False

# Mouse callback function
# def mouse_callback(event, x, y, flags, param):
#     global perform_ocr_flag
#     if event == cv.EVENT_RBUTTONDOWN:
#         perform_ocr_flag = True

# Set the mouse callback function
cv.namedWindow('Camera')
cv.setMouseCallback('Camera', draw_roi)

while(True): 

	ret, frame = cap.read()
      
	if ret:
        # Draw the ROI rectangle
		if roi is not None:
			roi_points = roi[0]
			cv.polylines(frame, [np.array(roi_points)], True, (0, 255, 0), 2)
                  
		cv.imshow('Camera', frame)
		# Check for key press or mouse event
		key = cv.waitKey(1) & 0xFF
            
		# Perform OCR if space key is pressed or right mouse button is clicked
		if key == ord(' ') or key == cv.EVENT_RBUTTONDOWN:
			# Perform OCR on the selected ROI
			if roi is not None:
				roi_points = roi[0]
				x1, y1 = roi_points[0]
				x2, y2 = roi_points[2]
				roi_img = frame[y1:y2, x1:x2]
			# Perform OCR on the captured frame
			digits_concat = digits_segmentaion(roi_img,
											config["inference"]["segmentation_threshold_value"],
											config["inference"]["segmentation_digit_min_area"],
											config["inference"]["segmentation_digit_crop_offset"])
			labels = ocr(digits_concat)
			print(labels)

			# Draw the recognized digits on the frame
			cv.putText(frame, str(labels), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

			# Display the frame with recognized digits
			cv.imshow('Camera', frame)

			# Reset the OCR flag
			perform_ocr_flag = False

		# Exit the loop if 'q' is pressed
		if key == ord('q'):
			break		
	else:
		break
cap.release() 
cv.destroyAllWindows() 