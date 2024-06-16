from utils import read_config
import cv2 as cv


if __name__ == '__main__':
    config = read_config()
    image = cv.imread(config["inference"]["images_path"] + "numbers_1.jpg")
    cv.imshow("image", image); cv.waitKey(0); cv.destroyAllWindows()