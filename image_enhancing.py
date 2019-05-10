import cv2
import math
import numpy as np
from matplotlib import pyplot as plt


def image_enhancement():
    exp1('res/moon.tif')
    exp1('res/fruits.bmp')
    exp2('res/moon.tif')
    exp3('res/moon.tif')
    exp3('res/fruits.bmp')
    exp4('res/moon.tif')
    exp4('res/fruits.bmp')
    exp5('res/moon.tif')
    exp5('res/fruits.bmp')
    exp6('res/moon.tif')
    exp6('res/fruits.bmp')


def exp1(str):  # 负像变换
    src = cv2.imread(str)
    dst = 255 - src
    res = np.hstack((src, dst))
    cv2.imshow('Negative conversion', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    # 读取文件
    src = cv2.imread(str)
    # 获取图片的大小
    img_info = src.shape
    image_height = img_info[0]
    image_weight = img_info[1]
    # 新建一个空图像用于存储计算结果
    dst = np.zeros((image_height, image_weight, 3), np.uint8)
    # 对每一个通道的值
    for i in range(image_height):
        for j in range(image_weight):
            (b, g, r) = src[i][j]
            dst[i][j] = (255 - b, 255 - g, 255 - r)
    # 将原图像和变换图像并排
    res = np.hstack((src, dst))
    # 显示图形，等待键盘消息后退出
    cv2.imshow('Negative conversion', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''


def exp2(str):  # 线性非对称变换
    src = cv2.imread(str)
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
    res = np.hstack((src, dst1, dst2, dst5))
    cv2.imshow('Log conversion (o, c=1, c=0.2, c=5)', res)
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
    res = np.hstack((src, dst1, dst2, dst5))
    cv2.imshow('Gamma conversion (o, gamma=1, gamma=0.2, gamma=5)', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def exp3(str):  # 邻域平均
    src = cv2.imread(str)
    dst3 = cv2.blur(src, (3, 3))
    dst7 = cv2.blur(src, (7, 7))
    res = np.hstack((src, dst3, dst7))
    cv2.imshow('Adjacent average conversion (o, 3x3, 7x7)', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def exp4(str):  # 中值平均
    src = cv2.imread(str)
    dst3 = cv2.blur(src, (7, 7))
    dst7 = cv2.medianBlur(src, 7)
    res = np.hstack((src, dst3, dst7))
    cv2.imshow('Average conversion (o, a, m)', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def exp5(str):  # Laplace 锐化
    src = cv2.imread(str, 0)
    dst = cv2.Laplacian(src, cv2.CV_16S, ksize=3)
    dst = cv2.convertScaleAbs(dst)
    res = np.hstack((src, dst))
    cv2.imshow('Laplace conversion', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    src = cv2.imread(str)
    weight = src.shape[0]
    height = src.shape[1]
    t1 = list([[0, 1, 0], [1, -4, 1], [0, 1, 0]])  # Laplace 滤波器
    dst = np.pad(src, ((1, 1), (1, 1), (0, 0)), "constant", constant_values=0)  # 边界问题处理
    for i in range(1, weight - 1):
        for j in range(1, height - 1):
            dst[i, j] = abs(np.sum(dst[i: i + 3, j:j + 3] * t1))
    dst = src + dst[1:dst.shape[0] - 1, 1:dst.shape[1] - 1]  # 滤波加成
    res = np.hstack((src, dst))
    cv2.imshow('Laplace conversion', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''


def exp6(str):  # Equalized Hist
    src = cv2.imread(str, 0)
    dst = cv2.equalizeHist(src)
    res = np.hstack((src, dst))
    cv2.imshow('Equalized Hist', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    hist, bins = np.histogram(src.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    plt.subplot(1,2,1)
    plt.plot(cdf_normalized, color='b')
    plt.hist(src.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.subplot(1,2,2)
    hist, bins = np.histogram(dst.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(dst.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()
