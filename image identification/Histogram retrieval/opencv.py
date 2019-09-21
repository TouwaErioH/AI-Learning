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
img = cv.imread('E:/2019HW_1.jpg', 1)
cv.imwrite('E:/2019HW_1.bmp', img)
'''

#jpg和bmp转为灰度图
'''
img1 = cv.imread('E:/2019HW_1.jpg', 1)
img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
cv.imwrite('E:/2019HW_1_gray.jpg', img1)

img2 = cv.imread('E:/2019HW_1.bmp', 1)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
cv.imwrite('E:/2019HW_1_gray.bmp', img2)
'''

'''
质量压缩
for num in range(0,6):
    img = cv.imread('E:/2019HW_1.bmp', 1)
    start = time.time()
    cv.imwrite('E:/lossy'+str(num)+'.jpg',img,[int(cv.IMWRITE_JPEG_QUALITY),100-num*10])
    end = time.time()
    print('lossy %d use %.5f s' %(100-num*10,end-start))
'''


# bmp格式RGB各8位，将字符串转化为二进制串后，每bit选取一个像素点的R层最后一位,变为str的二进制串


#读文件中字符串 ord 字符转ascii bin转二进制 不足8bits的补全8bits。每个字符占8bits，8pixels，lenth+8
#lenth为信息总bit数
f=open('E:/names.txt','r',encoding='utf-8')
res=f.read()
print(res)
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
img1 = cv.imread('E:/2019HW_1.bmp', 1)
size=img1.shape
width=size[1]
height=size[0]

#随机选取lenth个像素用于隐藏信息
x=random.sample(range(1,width-1),lenth);  #选取了pixels的x,y坐标,不重复;但是注意img是先高后宽，所以操作像素点时要注意
y=random.sample(range(1,height-1),lenth);

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
   
bimg=cv.split(img1)[0]
gimg=cv.split(img1)[1]



merged=cv.merge([bimg,gimg,hideimg])
cv.imwrite('E:/merged.bmp', merged)

img1 = cv.imread('E:/merged.bmp', 1)    
readimg=cv.split(img1)[2]
output=''
rang=len(res)

#恢复信息
for i in range(0,rang):
    result=''
    for j in range(0,8):
        k=i*8+j
        result=result+bin(readimg[y[k],x[k]])[-1]
    output=output+chr(int(result,2))
#final=result.decode('utf-8')
print(output)

