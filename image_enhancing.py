import cv2
import math
import numpy as np


def image_enhancement():
    exp1()
    exp2()


def exp1():  # 负像变换
    src = cv2.imread('res/moon.tif')
    img_info = src.shape
    h = img_info[0]
    w = img_info[1]
    dst = np.zeros((h, w, 3), np.uint8)
    for i in range(h):
        for j in range(w):
            (b, g, r) = src[i][j]
            dst[i][j] = (255 - b, 255 - g, 255 - r)
    cv2.imshow('Original Film', src)
    cv2.imshow('Negative Film', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    更简单的方法----直接使用255减得到负片
    src = cv2.imread('res/moon.tif')
    dst = 255 - src
    cv2.imshow('Original Film', src)
    cv2.imshow('Negative Film', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''


def exp2():  # 线性非对称变换
    src = cv2.imread('res/moon.tif')
    # 对数变换
    img_info = src.shape
    h = img_info[0]
    w = img_info[1]
    dst1 = np.zeros((h, w, 3))
    dst2 = np.zeros((h, w, 3))
    dst5 = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            for k in range(3):
                dst1[i, j, k] = 1.0 * math.log(1.0 + src[i, j, k])
                dst2[i, j, k] = 0.2 * math.log(1.0 + src[i, j, k])
                dst5[i, j, k] = 5.0 * math.log(1.0 + src[i, j, k])
    cv2.normalize(dst1, dst1, 0, 255, cv2.NORM_MINMAX)
    dst1 = cv2.convertScaleAbs(dst1)
    cv2.normalize(dst2, dst2, 0, 255, cv2.NORM_MINMAX)
    dst2 = cv2.convertScaleAbs(dst2)
    cv2.normalize(dst5, dst5, 0, 255, cv2.NORM_MINMAX)
    dst5 = cv2.convertScaleAbs(dst5)
    cv2.imshow('Original Film', src)
    cv2.imshow('Log Film c=1', dst1)
    cv2.imshow('Log Film c=0.2', dst2)
    cv2.imshow('Log Film c=5', dst5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 指数变换
    img_info = src.shape
    h = img_info[0]
    w = img_info[1]
    dst1 = np.zeros((h, w, 3))
    dst2 = np.zeros((h, w, 3))
    dst5 = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            for k in range(3):
                dst1[i, j, k] = math.pow(1.0 + src[i, j, k], 1)
                dst2[i, j, k] = math.pow(1.0 + src[i, j, k], 0.2)
                dst5[i, j, k] = math.pow(1.0 + src[i, j, k], 5)
    cv2.normalize(dst1, dst1, 0, 255, cv2.NORM_MINMAX)
    dst1 = cv2.convertScaleAbs(dst1)
    cv2.normalize(dst2, dst2, 0, 255, cv2.NORM_MINMAX)
    dst2 = cv2.convertScaleAbs(dst2)
    cv2.normalize(dst5, dst5, 0, 255, cv2.NORM_MINMAX)
    dst5 = cv2.convertScaleAbs(dst5)
    cv2.imshow('Original Film', src)
    cv2.imshow('Gamma Film γ=1, c=1', dst1)
    cv2.imshow('Gamma Film γ=0.2, c=1', dst2)
    cv2.imshow('Gamma Film γ=5, c=1', dst5)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
