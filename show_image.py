import cv2 as cv
import numpy as np


def show_image():
    src = cv.imread("res/Badge-01.png")
    cv.namedWindow("Image", cv.WINDOW_AUTOSIZE)
    cv.imshow("Image", src)
    get_image_info(src)
    change_gray(src)
    cv.waitKey(0)
    cv.destroyAllWindows()


def get_image_info(image):
    print(type(image))
    print(image.shape)
    print(image.size)
    print(image.dtype)
    pixel_data = np.array(image)
    print(pixel_data)


def change_gray(src):
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    cv.imwrite("res/result.png", gray)
    cv.waitKey(0)
    cv.destroyAllWindows()