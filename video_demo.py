import cv2 as cv


def video_demo():
    capture = cv.VideoCapture("res/pre-demo.mp4")
    flip = False
    while True:
        ret, frame = capture.read()
        if flip:
            frame = cv.flip(frame,1)
        cv.imshow("video", frame)
        c = cv.waitKey(30)
        if c == 13:
            flip = not flip
        if c == 27:
            break
    cv.destroyAllWindows()