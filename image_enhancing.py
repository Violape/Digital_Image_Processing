import cv2
import math
import numpy as np


def image_enhancement():
    # exp1()
    # exp2()
    # exp3()
    # exp4()
    exp5()


def exp1():  # 负像变换
    src = cv2.imread('res/moon.tif')
    dst = 255 - src
    cv2.imshow('Original Film', src)
    cv2.imshow('Negative Film', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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


def exp3():  # 邻域平均
    src = cv2.imread('res/moon.tif')
    dst3 = cv2.blur(src, (3, 3))
    dst7 = cv2.blur(src, (7, 7))
    cv2.imshow('Original Film', src)
    cv2.imshow("Blur range=3", dst3)
    cv2.imshow("Blur range=7", dst7)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def exp4():  # 中值平均
    src = cv2.imread('res/moon.tif')
    dst = cv2.medianBlur(src, 3)
    cv2.imshow('Original Film', src)
    cv2.imshow("Median range=3", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def exp5():  # Laplace 锐化
    src = cv2.imread('res/moon.tif')
    weight = src.shape[0]
    height = src.shape[1]
    t1 = list([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # Laplace 滤波器
    dst = np.pad(src, ((1, 1), (1, 1), (0, 0)), "constant", constant_values=0)  # 边界问题处理
    for i in range(1, weight - 1):
        for j in range(1, height - 1):
            dst[i, j] = abs(np.sum(dst[i: i + 3, j:j + 3] * t1))
    cv2.imshow('Original Film', src)
    cv2.imshow("Laplace Sharpening", src + dst[1:dst.shape[0] - 1, 1:dst.shape[1] - 1])  # 滤波加成
    cv2.waitKey(0)
    cv2.destroyAllWindows()
