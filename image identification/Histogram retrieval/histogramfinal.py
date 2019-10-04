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

def search(imggroup,gsize,hisstart,hisend):
    corr=[]
    wron=[]         #corr记录正确查找个数(认为属于i组且确实属于i组)，wron记录错误个数(认为属于i组但不属于i组)
    dis=[]                #dis记录认为属于某组的图片的距离该组的距离;如dis[1]记录所有认为属于第二组的图片的距离该组的距离
    result=[]
    index=[]
    for i in range(0,gsize+1):
        dis.append([])
        corr.append(0)
        wron.append(0)
        index.append([])
    ghis=[]
    gstart=[]
    gend=[]
    for i in range(0,gsize):         #ghis 记录各个选取图片的histogram gstart记录图片所属组的开始index
        ghis.append(histo[imggroup[i][0]])
        gstart.append(imggroup[i][1])
        gend.append(imggroup[i][2])
    #print(gend,gstart)
    for i in range(hisstart,hisend+1):  #计算图片距离所选取的每幅图片距离，选择最近的认为属于该组;cnt记录距离各组距离
        cnt=[]
        for j in range(0,gsize):
            temp=histo[i]-ghis[j]
            temp=np.power(temp,2)
            tempdis=np.sum(temp)
            cnt.append(tempdis)
        tpp=np.argsort(cnt)
        final=tpp[0]                               #img i 距离final组最近，认为属于final组
        #if 100<i&i<150:
        #   print(tpp[0])
        #  print(cnt)
        dis[final].append(cnt[final])
        index[final].append(i)
        
    for i in range(0,gsize): 
        corr=0
        wron=0
        result.append(np.argsort(dis[i]))                 #argsort 得到的result是dis升序排序后的索引序列
        for j in range(0,result[i].size):                 #result[0]是被search的图片自身，imgnum
            if (gstart[i]<=index[i][result[i][j]])&(index[i][result[i][j]]<=gend[i]):
                corr=corr+1
            else:
                wron=wron+1
        close=[]
        for k in range(0,6):
            close.append(index[i][result[i][k]])
        pre=corr/(wron+corr)
        recall=corr/100
        print(" imgnumber %d 's precision:%.5f ,recall:%.5f"%(imggroup[i][0],pre,recall))
        print(close)


if __name__ == '__main__': 
    #jpg2bmp()
    #histo=calcRGBhisto(0,9907,32)
    histostart = 0
    histoend = 500
    histobins=32
    histo = calcRGBhisto(histostart,histoend,histobins)
    #group=[[50,0,99],[150,100,199],[350,300,399],[750,700,799],[850,800,899],[950,899,998],
    #      [1150,1106,1205],[8700,8641,8740],[9050,9029,9128],[1650,1593,1692]]
    group=[[50,0,99],[137,100,199],[350,300,399]]
    groupsize=3
    search(group,groupsize,histostart,histoend)

