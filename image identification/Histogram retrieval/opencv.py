# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:20:25 2019

@author: erio
"""

import cv2 as cv
import random
import time

#将jpg转换为bmp

'''
start = time.time()
img = cv.imread('E:/2019HW_1.jpg', 1)
cv.imwrite('E:/2019HW_1.bmp', img)
end = time.time()
print('JPG to BMP use %.5f s' %(end-start))


#jpg和bmp转为灰度图

start = time.time()
img1 = cv.imread('E:/2019HW_1.jpg', 1)
img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
cv.imwrite('E:/2019HW_1_gray.jpg', img1)
end = time.time()
print('JPG to GRAY use %.5f s' %(end-start))

start = time.time()
img2 = cv.imread('E:/2019HW_1.bmp', 1)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
cv.imwrite('E:/2019HW_1_gray.bmp', img2)
end = time.time()
print('BMP to GRAY use %.5f s' %(end-start))
'''

'''
#质量压缩
for num in range(0,9):
    img = cv.imread('E:/2019HW_1.bmp', 1)
    na=100-10*num
    start = time.time()
    cv.imwrite('E:/lossy'+str(na)+'.jpg',img,[int(cv.IMWRITE_JPEG_QUALITY),100-num*10])
    end = time.time()
    print('lossy %d use %.5f s' %(100-num*10,end-start))

'''

# bmp格式RGB各8位，将字符串转化为二进制串后，每bit选取一个像素点的R层最后一位,变为str的二进制串

# text为包含输入字符串的txt文件，src为原始bmp，output为隐藏信息的bmp文件名
#return list x,y为隐藏了信息的pixels坐标
def embed(text,src,output):
    #open()读文件中字符串 ord()将字符转ascii bin()转二进制 不足8bits的补全8bits。之后存储到strbin
    #strbin存储转换为8位二进制ascii的信息，lenth为strbin长度，也即信息总bit数
    f=open(text,'r',encoding='utf-8')
    res=f.read()
    #print(res)
    strbin=''
    lenth=0
    for c in res:
        k=bin(ord(c)).replace('0b', '')
        if len(k)<8:
            k='0'*(8-len(k))+k
        strbin=strbin+k
        lenth=lenth+8
    #strbin=strbin+'s'+(bin(ord(c)).replace('0b', ''))
    f.close()
    #读图并获得高度宽度
    img1 = cv.imread(src, 1)
    start = time.time()
    size=img1.shape
    width=size[1]
    height=size[0]
    #随机选取lenth个像素用于隐藏信息
    x=random.sample(range(1,width-1),lenth);  #选取了pixels的x,y坐标,不重复;
    y=random.sample(range(1,height-1),lenth);
    #print(type(x))                          #隐藏了信息的像素点坐标存在list x,y中
    # 通道分离，注意顺序BGR不是RGB
    #(B, G, R) = cv2.split(image)
    #提取R通道，在所选像素最后一位写入信息
    hideimg=cv.split(img1)[2]  
    for i in range(0,lenth):
        #print(hideimg[y[i],x[i]])
        #print(bin(hideimg[y[i],x[i]]))
        tmp=bin(hideimg[y[i],x[i]])
        tmp=tmp[0:-1]+strbin[i]
        #print(tmp)
        #print(strbin[i])
        rep=int(tmp,2)
        #print(rep)
        hideimg[y[i],x[i]]=rep   
    bimg=cv.split(img1)[0]               #src的B通道
    gimg=cv.split(img1)[1]
    # 将写入信息的R通道和原始BG通道合并
    merged=cv.merge([bimg,gimg,hideimg])
    cv.imwrite(output, merged)
    end = time.time()
    #print('use %.5f s' %(end-start))
    return x,y

#hidebmp为隐藏了信息的文件名 list x，y为隐藏了信息的pixels坐标
def extract(hidebmp,x,y):
    img1 = cv.imread('E:/merged.bmp', 1)    
    start = time.time()
    readimg=cv.split(img1)[2]
    output=''
    rang=int(len(x)/8) #隐藏信息字符个数
    #恢复信息
    for i in range(0,rang):
        result=''
        for j in range(0,8):
            k=i*8+j
            result=result+bin(readimg[y[k],x[k]])[-1]
        output=output+chr(int(result,2))
        #final=result.decode('utf-8')
    end = time.time()
    #print('use %.5f s' %(end-start))
    print(output)


if __name__ == '__main__': 
    message='E:/names.txt'   #包含信息的TXT文件
    srcbmp='E:/2019HW_1.bmp' #原始bmp文件
    outputbmp='E:/merged.bmp' #隐藏了信息的bmp文件名
    x,y=embed(message,srcbmp,outputbmp)
    extract(outputbmp,x,y)
    