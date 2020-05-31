import cv2
import time
import numpy as np


def getHProjection(image):
    '''水平投影'''
    hProjection = np.zeros(image.shape, np.uint8)
    # 图像高与宽
    (h, w) = image.shape
    # 长度与图像高度一致的数组
    h_ = [0]*h
    # 循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y, x] == 255:
                h_[y] += 1
                # 绘制水平投影图像
    # for y in range(h):
    #     for x in range(h_[y]):
    #         hProjection[y, x] = 255
    # cv2.imwrite('hProjection2.jpg', hProjection)
    return h_


def getWProjection(image):
    '''竖直投影'''
    wProjection = np.zeros(image.shape, np.uint8)
    # 图像高与宽
    (h, w) = image.shape
    # 长度与图像高度一致的数组
    w_ = [0]*w
    # 循环统计每一行白色像素的个数
    for y in range(w):
        for x in range(h):
            if image[x, y] == 255:
                w_[y] += 1
                # 绘制水平投影图像
    # for y in range(w):
    #     for x in range(w_[y]):
    #         wProjection[x, y] = 255
    # cv2.imwrite('wProjection2.jpg',wProjection)
    # image_name = 'wProjection_'+image_name
    # cv2.imwrite(image_name, wProjection)
    return w_


def character_to_word(W_Start, W_End):
    """
    将w的切分变为以字为单位进行切分
    """
    W_word_Start = []
    W_word_End = []
    for i in range(len(W_Start)-1):
        if W_Start[i+1] - W_End[i] > 20:
            W_word_Start.append(W_Start[0])
            W_word_Start.append(W_Start[i+1])
            W_word_End.append(W_End[i])
            W_word_End.append(W_End[-1])
            break
    return W_word_Start, W_word_End


def crop_img(image_name):
    # 读入原始图像
    # image_name = '1965 Q4.jpg'
    origineImage = cv2.imread(image_name)
    # 图像灰度化
    #image = cv2.imread('test.jpg',0)
    image = cv2.cvtColor(origineImage, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray',image)
    # 将图片二值化
    retval, img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    # cv2.imwrite('binary.jpg', img)
    # 图像高与宽
    (h, w) = img.shape
    # 水平投影
    H = getHProjection(img)

    start = 0
    H_Start = []
    H_End = []
    W_Start = []
    W_End = []
    # 根据水平投影获取垂直分割位置
    for i in range(len(H)):
        if H[i] > 0 and start == 0:
            H_Start.append(i)
            start = 1
        if H[i] <= 0 and start == 1:
            H_End.append(i)
            start = 0
    # 分割行，分割之后再进行列分割并保存分割位置
    for i in range(len(H_Start)):
        # 设定阈值排除噪声
        if H_End[i] - H_Start[i] < 5:
            continue
        start = 0
        W_Start = []
        W_End = []
        # 获取行图像
        cropImg = img[H_Start[i]-3:H_End[i]+3, :]
        # cropImg_name = image_name + '_' + str(i) + '_row.jpg'
        # print(cropImg_name)
        # cv2.imwrite(cropImg_name, cropImg)
        # 竖直投影
        W = getWProjection(cropImg)
        for j in range(len(W)):
            if W[j] > 0 and start == 0:
                W_Start.append(j)
                start = 1
            if W[j] <= 0 and start == 1:
                W_End.append(j)
                start = 0
        # print('W_Start:'+str(len(W_Start)))
        W_word_Start, W_word_End = character_to_word(W_Start, W_End)
        for j in range(len(W_word_Start)):
            crop_w_Img = cropImg[:, W_word_Start[j]-3:W_word_End[j]+3]
            cropImg_w_name = image_name + '_' + \
                str(i) + '_row_' + str(j) + '_line.jpg'
            # print(cropImg_w_name)
            cv2.imwrite(cropImg_w_name, crop_w_Img)
    

    # https://www.jb51.net/article/164611.htm


if __name__ == '__main__':
    start_time = time.time()
    for k in range(55):
        image_name = str(1965 + k) + '.jpg'
        crop_img(image_name)
    end_time = time.time()
    print(start_time-end_time)
