import cv2 as cv
import numpy as np


def show(image):
    cv.imshow("image", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

def contour_center(contours):
    cx = []; cy = []
    cnt = 0
    for c in contours:
        M = cv.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"]); cx.append(cX)
            cY = int(M["m01"] / M["m00"]); cy.append(cY)
            cnt += 1
    return cx, cy

def contour_sorted(contours):
    cx, cy = contour_center(contours)
    sort = sorted(cx, reverse=False)
    sorted_contours = []
    for c in sort:
        sorted_contours.append(contours[cx.index(c)])
    return sorted_contours

def contour_filter(contours, min_area):
    filtered_contours = [cnt for cnt in contours if cv.contourArea(cnt) >= min_area]
    return filtered_contours

def digits_crop(image, height, width, contours, offset, save_digits=False):
    num = 0
    digits = []
    for cnt in contours:
        
        # Get bounding box for contour
        x, y, w, h = cv.boundingRect(cnt)
        # Adjust the bounding rectangle with the offset
        x -= offset
        y -= offset
        w += 2 * offset
        h += 2 * offset
        
        # Ensure the bounding rectangle stays within the image boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)        
        # Crop contour
        crop = image[y:y+h, x:x+w]
        image_gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
        _, image_threshold = cv.threshold(image_gray, 150, 255, cv.THRESH_BINARY_INV)        
        # Get dimensions for saving
        # height, width = crop.shape[:2]
        
        final_img = cv.resize(image_threshold, (height, width))
        digits.append(final_img)
        # Save to file
        if save_digits:
            cv.imwrite(f'output_{num}.jpg', final_img, [cv.IMWRITE_JPEG_QUALITY, 10])
        num += 1
    return digits

def contour_mins(contours):
    contours_min_location = []  
    for cnt in contours:
        # Initialize min_x and min_y with large values
        min_x, min_y = float('inf'), float('inf')
        min_x_list = []; min_y_list = []
        
        # Iterate over the contour points
        for point in cnt:
            x, y = point[0]
            
            # Update min_x and min_y
            min_x = min(min_x, x); min_x_list.append(min_x)
            min_y = min(min_y, y); min_y_list.append(min_y)
            
        contours_min_location.append([min(min_x_list), min(min_y_list)])
    return contours_min_location
   
def digits_segmentaion(image, height=100, width=100, threshold_value=150, digit_min_area=100, digit_crop_offset=10):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, image_threshold = cv.threshold(image_gray, threshold_value, 255, cv.THRESH_BINARY_INV)
    # Find contours
    contours, _ = cv.findContours(image_threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filtered_contours = contour_filter(contours, min_area=digit_min_area)
    sorted_contours = contour_sorted(filtered_contours)
    digits = digits_crop(image, height, width, sorted_contours, offset=digit_crop_offset)
    digits_concat = np.stack(digits)
    
    contours_min_location = contour_mins(sorted_contours)

    return digits_concat, contours_min_location

if __name__ == '__main__':
    from utils import read_config
    config = read_config()
    image = cv.imread(config["inference"]["images_path"] + "numbers_1.jpg")
    segmented_digits, contours_min_location = digits_segmentaion(image)
    print(segmented_digits.shape)
    print(contours_min_location)
    for i in range(10):
        show(segmented_digits[i])


