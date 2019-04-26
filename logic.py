import cv2 as cv
import numpy as np


def add(m1, m2):
    dst = cv.add(m1, m2)
    cv.imshow("Add", dst)


def subtract(m1, m2):
    dst = cv.subtract(m1, m2)
    cv.imshow("Subtract I", dst)
    dst = cv.subtract(m2, m1)
    cv.imshow("Subtract II", dst)


def logics(m1, m2):
    dst = cv.bitwise_and(m1, m2)
    cv.imshow("And", dst)
    dst = cv.bitwise_or(m1, m2)
    cv.imshow("Or", dst)
    dst = cv.bitwise_not(m1)
    cv.imshow("Not", dst)


def contrast(image, c, b):
    h, w, ch = image.shape
    blank = np.zeros([h, w, ch], image.dtype)
    dst = cv.addWeighted(image, c, blank, 1 - c, b)
    cv.imshow("Contrast", dst)


def logic():
    src1 = cv.imread("res/Badge-01.png")
    src2 = cv.imread("res/Badge-02.png")
    print(src1.shape)
    print(src2.shape)

    contrast(src1, 1.5, 0)
    cv.waitKey(0)
    cv.destroyAllWindows()

    add(src1, src2)
    subtract(src1, src2)
    cv.waitKey(0)
    cv.destroyAllWindows()

    img = cv.imread("res/Badge-02.png")
    logics(src1, img)
    cv.waitKey(0)
    cv.destroyAllWindows()
