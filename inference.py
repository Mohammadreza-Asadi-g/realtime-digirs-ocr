from src.segmentation import digits_segmentaion
from src.ocr import ocr
import cv2 as cv

# Using Phone Camera
cap  = cv.VideoCapture('http://192.168.0.35:4747/video') 

if not cap.isOpened():
    print("Cannot open camera!")
    exit()
	
while(True): 

	ret, frame = cap.read()
	if ret:
		cv.imshow('frame', frame) 
	
	if cv.waitKey(1) & 0xFF == ord('q'): 
		break

cap.release() 
cv.destroyAllWindows() 