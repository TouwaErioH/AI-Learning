# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:01:42 2019

@author: erio
"""
'''
从下面10组中随机各选取一幅图片,
直接根据其颜色直方图在9908(0-9907)幅图片中按照最邻近方法找出最接近的图片(若超出100幅取最接近的100幅。若不足100幅则不足)
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
import time

#转bmp
def jpg2bmp():
    for i in range(0,9908):
        nam='D:/image/'+str(i)+'.jpg'
        nam1='D:/image/'+str(i)+'.bmp'
        img = cv2.imread(nam, 1)
        cv2.imwrite(nam1, img)

#计算图片的hsv histogram h,s为Hue, Saturation的bins
def calchsv(start,end,h,s):
    histoh=[]       
    for i in range(start,end+1):
        img = cv2.imread('D:/image/'+str(i)+'.bmp')
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        hist1 = cv2.calcHist([hsv], [0, 1], None, [h, s], [0, 180, 0, 256])
        #cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX, -1);    #归一化，后面数据处理的方便，其次是保证程序运行时收敛加快
        histoh.append(hist1)
    return histoh

#计算图片的rgb histogram 。分三个通道RGB计算，每个通道bins为bins，而后拼接为一个histogram.
def calcRGBhisto(start,end,bins):
    histor=[]                                          #histo storage all histogram nparrays
    for i in range(start,end+1):    
        img = cv2.imread('D:/image/'+str(i)+'.bmp') 
        #histc = cv2.calcHist([img],[0,1,2],None,[bins,bins,bins],[0,255,0,255,0,255])
        hist2 = cv2.calcHist([img],[2],None,[bins],[0,255])  #np.array   RGB三通道 
        hist0 = cv2.calcHist([img],[0],None,[bins],[0,255])  #np.array
        hist1 = cv2.calcHist([img],[1],None,[bins],[0,255])  #np.array
        histf=np.vstack((hist0,hist1,hist2))            #histf,当前图片的直方图的nparray
        #cv2.normalize(histf, histf, 0, 1, cv2.NORM_MINMAX, -1);
        histor.append(histf)
    return histor


#imggroup为记录搜索数据的array。形如[[50,0,99],[150,100,199]] 意味第一组图片范围为0-99，选取50号图片为标准搜索
#hisstart,hisend为要做检索的图片范围，如0,9907
#method为采用的检索方式。如Euclid为根据图片的直方图的欧几里得距离，越短认为越接近
def search(imggroup,gsize,hisstart,hisend,method): #method记录判定方式，如Euclid;
    if (method=="Euclid"):
        calmethod="Euclid"
    elif (method=="Correlation"):
        calmethod=cv2.HISTCMP_CORREL
    elif (method=="ChiSquare"):
        calmethod=cv2.HISTCMP_CHISQR
    else:
        calmethod=cv2.HISTCMP_BHATTACHARYYA
    rgbdis=[]                #rgbdis记录认为属于某组的图片的距离该组标准的距离;如dis[1]记录所有认为属于第二组的图片的距离该组标准150.bmp的距离
    hsvdis=[]
    rgbresult=[]    #rgbresult记录结果,为gsize维数组。从0开始index越小越接近标准。如rgbresult[0]数组记录判定为属于1组的图片index。如rgbresult[0][0]=50，即为50.bmp
    hsvresult=[]
    rgbindex=[]     #rgbindex[0]数组记录判定为属于1组的图片index。
    hsvindex=[]
    for i in range(0,gsize+1):
        rgbdis.append([])
        hsvdis.append([])
        rgbindex.append([])
        hsvindex.append([])
    ghisrgb=[] #选取为标准的图片的histogram数组。1维数组。如ghisrgb[0]为50.bmp的histogram
    ghishsv=[]
    gstart=[] #记录每组开始图片index
    gend=[]
    for i in range(0,gsize):         #ghisrgb,ghishsv 记录各个选取图片的histogram gstart记录图片所属组的开始index，
        ghisrgb.append(historgb[imggroup[i][0]]) #如第一组0-99，查找50.bmp，gstart[0]=0，gend[0]=99，ghisrgb[0]=histo[50]
        ghishsv.append(histohsv[imggroup[i][0]])
        gstart.append(imggroup[i][1])
        gend.append(imggroup[i][2])
    for i in range(hisstart,hisend+1):  #在hisstart-hisend的范围内，计算图片距离所选取的每幅图片距离，选择最近的认为属于该组;cnt记录距离各组距离
        tprgbdis=[]
        tphsvdis=[]  #记录当前图片采用rgb和hsvhistogram,距离选取出的图片的距离
        for j in range(0,gsize):       
            if (calmethod=="Euclid"):
                tempr=historgb[i]-ghisrgb[j]
                tempr=np.power(tempr,2)
                tempdis=np.sum(tempr)
                tprgbdis.append(tempdis)
                temph=histohsv[i]-ghishsv[j]
                temph=np.power(temph,2)
                tempdis=np.sum(temph)
                tphsvdis.append(tempdis)
            else:
                match = cv2.compareHist(historgb[i],ghisrgb[j], calmethod)
                tprgbdis.append(match)
                match = cv2.compareHist(histohsv[i],ghishsv[j], calmethod)
                tphsvdis.append(match)
        #np.argsort()用于给数组排序，返回值为从小到大元素index的值.
        #假设一个数组a为[0,1,2,20,67,3],使用numpy.argsort(a),返回值应为[0,1,2,5,3,4]
        if(method=="Correlation"):  #Correlation -1-1,越大越接近
            tpp=np.argsort(tprgbdis)
            final=tpp[gsize-1]                 #img i 与final组correlation最接近1，认为属于final组
            rgbdis[final].append(tprgbdis[final]) #将i.img距离final组标准的距离记录到rgbdis[final]数组，同时其index i记录到rgbindex[final]数组。
            rgbindex[final].append(i)
            
            tpp=np.argsort(tphsvdis)
            final=tpp[0]                 #img i 与final组euclid距离最短，认为属于final组
            hsvdis[final].append(tphsvdis[final]) 
            hsvindex[final].append(i)
        else:   #其余的三种方法越小越接近
            tpp=np.argsort(tprgbdis)
            final=tpp[0]                 #img i 与final组距离最短，认为属于final组
            rgbdis[final].append(tprgbdis[final]) 
            rgbindex[final].append(i)
            
            tpp=np.argsort(tphsvdis)
            final=tpp[0]              
            hsvdis[final].append(tphsvdis[final]) 
            hsvindex[final].append(i)
        
        #gstart[i],第i组起始index。如第0组0-99，gstart[0]=0,gend[0]=99;
        #index[i][result[i][j]] index[i]为认为属于[i]组的图片的index数组.result[i]为认为属于i组的图片的dis升序排序后的索引序列
        #例如 index[0]={0,24,5} 那么result[0][2]=5
    for i in range(0,gsize): 
        rgbcorr=0 #corr 认为属于i组并且确实属于i组
        hsvcorr=0
        if(method=="Correlation"): #越大越好
            #print(type(rgbdis))
            nprd=np.array(rgbdis[i])
            nphd=np.array(hsvdis[i])
            rgbresult.append(np.argsort(-nprd))                 #argsort 得到的result是dis降序排序后的索引序列，数组
            hsvresult.append(np.argsort(-nphd))   #result数组中增加一个数组，如[]-->[[0,1,2]],判定属于第0组的图片index为0,1,2
        else: #越小越好
            rgbresult.append(np.argsort(rgbdis[i]))    
            hsvresult.append(np.argsort(hsvdis[i]))
        
        #print(rgbresult[i].size,hsvresult[i].size)
        if rgbresult[i].size<=100:   #若判定为i组的图片不足100幅
            rgbsize=rgbresult[i].size
        else:                       #若判定为i组的图片超过100幅,取最近的100幅
            rgbsize=100
        if hsvresult[i].size<=100:   #若判定为i组的图片不足100幅
            hsvsize=hsvresult[i].size
        else:                       #若判定为i组的图片超过100幅,取最近的100幅
            hsvsize=100
        for j in range(0,rgbsize):                 #result[i][0]理论上是被search的图片自身，imgnum
            if (gstart[i]<=rgbindex[i][rgbresult[i][j]])&(rgbindex[i][rgbresult[i][j]]<=gend[i]):
                rgbcorr=rgbcorr+1
        for j in range(0,hsvsize):                 #result[i][0]理论上是被search的图片自身，imgnum
            if (gstart[i]<=hsvindex[i][hsvresult[i][j]])&(hsvindex[i][hsvresult[i][j]]<=gend[i]):
                hsvcorr=hsvcorr+1
        rgbclose=[]
        hsvclose=[]   #取最接近的5幅图片展示
        for k in range(0,6):
            rgbclose.append(rgbindex[i][rgbresult[i][k]])
            hsvclose.append(hsvindex[i][hsvresult[i][k]])
        rgbpre=rgbcorr/(rgbsize)
        rgbrecall=rgbcorr/100
        hsvpre=hsvcorr/(hsvsize)
        hsvrecall=hsvcorr/100
        print("imgnumber %d\n"%(imggroup[i][0]))
        print("using rgb histogram %s method 's precision:%.5f ,recall:%.5f\n"%(method,rgbpre,rgbrecall))
        print("using hsv histogram %s method 's precision:%.5f ,recall:%.5f\n"%(method,hsvpre,hsvrecall))
        print("rgb close 5\n")
        print(rgbclose)
        print("hsv close 5\n")
        print(hsvclose)
        

if __name__ == '__main__': 
    #jpg2bmp()
    #histo=calcRGBhisto(0,9907,32)
    histostart = 0
    histoend = 500
    histobins=32
    h= 30  #180
    s= 128  #256
    historgb = calcRGBhisto(histostart,histoend,histobins)
    histohsv = calchsv(histostart,histoend,h,s)
    #每组中选取一幅图片，查找最近的100张，计算查全率查准率。如第一组0-99，查50.bmp
    #group=[[50,0,99],[150,100,199],[350,300,399],[750,700,799],[850,800,899],[950,899,998],
    #           [1150,1106,1205],[8700,8641,8740],[9050,9029,9128],[1650,1593,1692]]
    group=[[50,0,99],[137,100,199],[350,300,399]]
    groupsize = 3
    search(group,groupsize,histostart,histoend,"Euclid")
    #search(group,groupsize,histostart,histoend,"Correlation")
    #search(group,groupsize,histostart,histoend,"ChiSquare")
    #search(group,groupsize,histostart,histoend,"Bhattach")