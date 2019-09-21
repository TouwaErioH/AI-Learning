# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:20:25 2019

@author: erio
"""
'''
0-99 butterfly 100  mark 1,第一组图片
100-199 mountain 100 mark 2
700-799 luori 100  mark 3
#800-898 flower 99  unused
899-998 tree  100 mark 4
1593-1692 saiche 100 mark 5
'''
import cv2
import numpy as np
import time


#转换为RGB图
def jpg2bmp():
    for i in range(0,200):
        nam='E:/test/'+str(i)+'.jpg'
        nam1='E:/test/'+str(i)+'.bmp'
        img = cv2.imread(nam, 1)
        cv2.imwrite(nam1, img)
    for i in range(700,999):
        nam='E:/test/'+str(i)+'.jpg'
        nam1='E:/test/'+str(i)+'.bmp'
        img = cv2.imread(nam, 1)
        cv2.imwrite(nam1, img)    
    for i in range(1593,1693):
        nam='E:/test/'+str(i)+'.jpg'
        nam1='E:/test/'+str(i)+'.bmp'
        img = cv2.imread(nam, 1)
        cv2.imwrite(nam1, img)

'''
cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate ]]) ->hist
imaes:输入的图像
channels:选择图像的通道
mask:掩膜，是一个大小和image一样的np数组，其中把需要处理的部分指定为1，不需要处理的部分指定为0，一般设置为None，表示处理整幅图像
histSize:使用多少个bin，一般为256（即直方图的柱子数量）
ranges:像素值的范围，一般为[0,255]表示0~255
'''
#创建中心histogram  采用取已经分类好的5类中前20幅图的直方图均值计算 euclidean distance
#之后鉴别图片时，取最近的中心所在组作为鉴别组
#first用做中心的第一张图编号，sta第二个图片编号，end结束的图片编号bins直方图bins

#RGB计算直方图中心
#做完之后才发现可以直接一次计算三个通道，但是懒得改了，可见HSV图的例子
def calccenter(first,star,end,bins):
    img = cv2.imread('E:/test/'+str(first)+'.bmp')
    hist1_2 = cv2.calcHist([img],[2],None,[bins],[0,255])  #np.array
    hist1_0 = cv2.calcHist([img],[0],None,[bins],[0,255])  #np.array
    hist1_1 = cv2.calcHist([img],[1],None,[bins],[0,255])  #np.array
    for i in range(star,end):
        nam1='E:/test/'+str(i)+'.bmp'
        img = cv2.imread(nam1)
        hist2 = cv2.calcHist([img],[2],None,[bins],[0,255])
        hist0 = cv2.calcHist([img],[0],None,[bins],[0,255])
        hist1 = cv2.calcHist([img],[1],None,[bins],[0,255])
        hist1_1=hist1_1+hist1
        hist1_0=hist1_0+hist0
        hist1_2=hist1_2+hist2
    hist1_2=np.divide(hist1_2,20)
    hist1_1=np.divide(hist1_1,20)
    hist1_0=np.divide(hist1_0,20)
    histf=np.vstack((hist1_0,hist1_1,hist1_2))
    return histf


#计算各组图片距离中心的欧式距离，鉴别所属组，并在corr和wron数组记录
#sta开始鉴别的图片编号，end结束鉴别的图片编号，mark图片真实所属组，bins直方图bins
def calcdis(sta,end,mark,bins):
    #cor 真实情况
    cor=mark
    for k in range(sta,end):
        nam1='E:/test/'+str(k)+'.bmp'
        img = cv2.imread(nam1)
        histc2 = cv2.calcHist([img],[2],None,[bins],[0,255])  #np.array
        histc1 = cv2.calcHist([img],[1],None,[bins],[0,255])  #np.array
        histc0 = cv2.calcHist([img],[0],None,[bins],[0,255])  #np.array
        histc=np.vstack((histc0,histc1,histc2))
        re=0
        temp1=histc-histf1
        temp1=np.power(temp1,2)
        tempdis1=np.sum(temp1)
        #print(tempdis1)
        re=1
        temp=tempdis1
        
        temp2=histc-histf2
        temp2=np.power(temp2,2)
        tempdis2=np.sum(temp2)
        #print(tempdis2)
        if tempdis2<temp:
            re=2
            temp=tempdis2
            
        temp3=histc-histf3
        temp3=np.power(temp3,2)
        tempdis3=np.sum(temp3)
        #print(tempdis3)
        if tempdis3<temp:
            re=3
            temp=tempdis3
                
        temp4=histc-histf4
        temp4=np.power(temp4,2)
        tempdis4=np.sum(temp4)
        #print(tempdis4)
        if tempdis4<temp:
            re=4
            temp=tempdis4
                    
        temp5=histc-histf5
        temp5=np.power(temp5,2)
        tempdis5=np.sum(temp5)
        #print(tempdis5)
        if tempdis5<temp:
                        re=5
                        temp=tempdis5
                        #print(re)
        #判断是否判断正确，记录到wron，corr数组
        if cor==re:
            corr[cor]=corr[cor]+1
        else:
            wron[re]=wron[re]+1
        #print(corr)
        #print(wron)

#计算5组图片各自的的precision(查准率)，recall(查全率),并输出

#在opencv里面，你用8bit的uchar无法表示超过255的数据，所以，opencv做了一个小小的技巧性处理，直接把H分量的值除以2。
#计算H,S方向直方图
def calchsv(first,star,end,h,s):
    img = cv2.imread('E:/test/'+str(first)+'.bmp')
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [h, s], [0, 180, 0, 256])
    for i in range(star,end):
        nam1='E:/test/'+str(i)+'.bmp'
        img = cv2.imread(nam1)
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        hist1 = cv2.calcHist([hsv], [0, 1], None, [h, s], [0, 180, 0, 256])
        hist=hist+hist1
    hist=np.divide(hist,20)
    #print(hist.shape)  (180,256)
    return hist

def clacdishsv(sta,end,mark,h,s):
    #cor 真实情况
    cor=mark
    for k in range(sta,end):
        nam1='E:/test/'+str(k)+'.bmp'
        img = cv2.imread(nam1)
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        histc = cv2.calcHist([hsv], [0, 1], None, [h, s], [0, 180, 0, 256])
        re=0
        temp1=histc-histf1hsv
        temp1=np.power(temp1,2)
        tempdis1=np.sum(temp1)
        #print(tempdis1)
        re=1
        temp=tempdis1
        
        temp2=histc-histf2hsv
        temp2=np.power(temp2,2)
        tempdis2=np.sum(temp2)
        #print(tempdis2)
        if tempdis2<temp:
            re=2
            temp=tempdis2
            
        temp3=histc-histf3hsv
        temp3=np.power(temp3,2)
        tempdis3=np.sum(temp3)
        #print(tempdis3)
        if tempdis3<temp:
            re=3
            temp=tempdis3
                
        temp4=histc-histf4hsv
        temp4=np.power(temp4,2)
        tempdis4=np.sum(temp4)
        #print(tempdis4)
        if tempdis4<temp:
            re=4
            temp=tempdis4
                    
        temp5=histc-histf5hsv
        temp5=np.power(temp5,2)
        tempdis5=np.sum(temp5)
        #print(tempdis5)
        if tempdis5<temp:
                        re=5
                        temp=tempdis5
                        #print(re)
        #判断是否判断正确，记录到wron，corr数组
        if cor==re:
            corr[cor]=corr[cor]+1
        else:
            wron[re]=wron[re]+1
        #print(corr)
        #print(wron)

def calcresult():
    recalll=0
    precisionn=0
    for i in range(1,6):
        recalll=recalll+corr[i]
        precisionn=corr[i]+wron[i]+precisionn
    for p in range(1,6):
        pre=corr[p]/(wron[p]+corr[p])
        rec=corr[p]/100
        print("%d mark's precision:%.2f ,recall:%.2f"%(p,pre,rec))

if __name__ == '__main__': 
    #corr[i]=1    #图片属于i组并且被鉴别为i组
    #wron[i]=1    #图片不属于i组但被鉴别为i组
    #记录数组
    corr=[0,0,0,0,0,0]
    wron=[0,0,0,0,0,0]
    
    #jpg2bmp() 
    '''
    #RGB histogram 鉴别 retrieval
    testbin=256 #选择计算直方图时的bins
    histf1 = calccenter(0,1,20,testbin)
    histf2 = calccenter(100,101,120,testbin)
    histf3 = calccenter(700,701,720,testbin)
    histf4 = calccenter(899,900,919,testbin)
    histf5 = calccenter(1593,1594,1613,testbin)
    #print(histf5.shape) 
    
    #根据计算得到的中心鉴别图片
    calcdis(0,100,1,testbin)
    calcdis(100,200,2,testbin)
    calcdis(700,800,3,testbin)
    calcdis(899,999,4,testbin)
    calcdis(1593,1693,5,testbin)
    
    calcresult()
    '''
    #HSV histogram鉴别
    start = time.time()
    histf1hsv=calchsv(0,1,20,180,256)
    histf2hsv=calchsv(100,101,120,180,256)
    histf3hsv=calchsv(700,701,720,180,256)
    histf4hsv=calchsv(899,900,919,180,256)
    histf5hsv=calchsv(1593,1594,1613,180,256)
    end = time.time()
    print('clac HSV center use %.5f s' %(end-start))
    
    clacdishsv(0,100,1,180,256)
    clacdishsv(100,200,2,180,256)
    clacdishsv(700,800,3,180,256)
    clacdishsv(899,999,4,180,256)
    clacdishsv(1593,1693,5,180,256)
    
    calcresult()