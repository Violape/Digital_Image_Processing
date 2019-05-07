import cv2
import numpy as np


def image_enhancement():
    exp1()


def exp1():  # 负像变换
    src = cv2.imread('res/moon.tif')
    img_info = src.shape
    image_height = img_info[0]
    image_weight = img_info[1]
    dst = np.zeros((image_height, image_weight, 3), np.uint8)
    for i in range(image_height):
        for j in range(image_weight):
            (b, g, r) = src[i][j]
            dst[i][j] = (255 - b, 255 - g, 255 - r)
    cv2.imshow('Original Film', src)
    cv2.imshow('Negative Film', dst)
    cv2.waitKey(0)
    '''
    更简单的方法----直接使用255减得到负片
    src = cv2.imread('res/moon.tif')
    dst = 255 - src
    cv2.imshow('Original Film', src)
    cv2.imshow('Negative Film', dst)
    cv2.waitKey(0)
    '''