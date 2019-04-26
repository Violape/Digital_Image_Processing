import cv2 as cv


def equal_hist():
    src = cv.imread("res/Badge-01.png")
    cv.namedWindow("Input", cv.WINDOW_AUTOSIZE)
    cv.imshow("Input", src)
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    cv.imshow("Gray", gray)
    dst = cv.equalizeHist(gray)
    cv.imshow("Equal Hist", dst)
    cv.waitKey(0)
    cv.destroyAllWindows()