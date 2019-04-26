import cv2 as cv
import numpy as np
def split():
    src = cv.imread("res/Badge-01.png")
    cv.imshow("Input", src)

    b, g, r = cv.split(src)
    zero = np.zeros(src.shape[:2], np.uint8)

    blue = cv.merge([b, zero, zero])
    green = cv.merge([zero, g, zero])
    red = cv.merge([zero, zero, r])

    cv.imshow("Blue", blue)
    cv.imshow("Green", green)
    cv.imshow("Red", red)

    cv.waitKey(0)
    cv.destroyAllWindows()