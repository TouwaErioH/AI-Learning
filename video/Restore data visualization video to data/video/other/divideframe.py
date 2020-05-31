import cv2
 
import numpy as np
 
  
 

def getHProjection(image):
    '''水平投影''' 
    hProjection = np.zeros(image.shape,np.uint8)
    #图像高与宽
    (h,w)=image.shape 
    #长度与图像高度一致的数组
    h_ = [0]*h
    #循环统计每一行白色像素的个数
    for y in range(h):
        for x in range(w):
            if image[y,x] == 255:
                h_[y]+=1
                #绘制水平投影图像
    for y in range(h):
        for x in range(h_[y]):
            hProjection[y,x] = 255
    #cv2.imshow('hProjection2',hProjection)
    return h_
 

def crop_cut(image_name):
    #origineImage = cv2.imread(image_name)
    # 图像灰度化 
    #image = cv2.imread('test.jpg',0)
    #image = cv2.cvtColor(origineImage,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray',image)
    # 将图片二值化
    image = cv2.imread(image_name)
    retval, img = cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
    #cv2.imshow('binary',img)
    #图像高与宽
    (h,w)=img.shape 
    #水平投影
    H = getHProjection(img)

    start = 0
    H_Start = []
    H_End = []
    #根据水平投影获取垂直分割位置
    for i in range(len(H)):
        if H[i] > 0 and start ==0:
            H_Start.append(i)
            start = 1
        if H[i] <= 0 and start == 1:
            H_End.append(i)
            start = 0
    #分割行，分割之后再进行列分割并保存分割位置
    for i in range(len(H_Start)):
        #获取行图像
        cropImg = img[H_Start[i]-10:H_End[i]+10, :]
        cropImg_name = image_name + '_' + str(i) + '_line.jpg' 
        print(cropImg_name)       
        cv2.imwrite(cropImg_name, cropImg)

    # https://www.jb51.net/article/164611.htm
if __name__ == '__main__':
    for i in range(55):
        crop_cut(str(1965 + i) + ' Q4.jpg')
