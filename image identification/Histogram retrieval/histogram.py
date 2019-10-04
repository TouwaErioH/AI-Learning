# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:20:25 2019

@author: erio
"""
'''
从下面10组中随机各选取一幅图片,
直接根据其颜色直方图在9908幅图片中按照最邻近方法找出100幅最接近的图片
计算查全率查准率
并展示最接近的五福图片
0-99 butterfly 100  mark 1,第一组图片
100-199 mountain 100 mark 2
700-799 luori 100  mark 3
800-899 花  mark6
300-399 mark9 
1106-1205 mark 10
8641 8740 mark8
9029 9128 mark 7
899-998 tree  100 mark 4
1593-1692 saiche 100 mark 5

'''
import cv2
import numpy as np

#计算所有图片的直方矩阵

#计算直方矩阵的欧几里得距离时,因为对应的最后都要/bins*3,故不除简化运算
def calcRGBhisto(start,end,bins):
    histo=[]                                          #histo storage all histogram nparrays
    for i in range(start,end+1):    
        img = cv2.imread('D:/image/'+str(i)+'.bmp') 
        #histc = cv2.calcHist([img],[0,1,2],None,[bins,bins,bins],[0,255,0,255,0,255])
        hist2 = cv2.calcHist([img],[2],None,[bins],[0,255])  #np.array   RGB三通道 
        hist0 = cv2.calcHist([img],[0],None,[bins],[0,255])  #np.array
        hist1 = cv2.calcHist([img],[1],None,[bins],[0,255])  #np.array
        histf=np.vstack((hist0,hist1,hist2))            #histf,当前图片的直方图的nparray
        histo.append(histf)
    return histo

#转换为RGB图
def jpg2bmp():
    for i in range(0,9908):
        nam='D:/image/'+str(i)+'.jpg'
        nam1='D:/image/'+str(i)+'.bmp'
        img = cv2.imread(nam, 1)
        cv2.imwrite(nam1, img)

def search(imgnum,start,end,hisstart,hisend):
    corr=0
    wron=0         #corr记录正确查找个数，wron记录错误个数
    dis=[]
    histcenter=histo[imgnum]
    for i in range(hisstart,hisend+1):
        temp=histo[i]-histcenter
        temp=np.power(temp,2)
        tempdis=np.sum(temp)
        dis.append(tempdis)
    result=np.argsort(dis)                 #argsort 得到的result是dis升序排序后的索引序列
    for i in range(1,101):                 #result[0]是被search的图片自身，imgnum
        if (start<=result[i])&(result[i]<=end):
            corr=corr+1
        else:
            wron=wron+1
    close5=result[1:6]
    pre=corr/(wron+corr)
    recall=corr/100
    print(" imgnumber %d 's precision:%.2f ,recall:%.2f"%(imgnum,pre,recall))
    print(close5)


if __name__ == '__main__': 
    #jpg2bmp()
    #histo=calcRGBhisto(0,9907,32)
    histostart = 0
    histoend = 500
    histobins=32
    histo = calcRGBhisto(histostart,histoend,histobins)
    search(106,100,199,histostart,histoend)
