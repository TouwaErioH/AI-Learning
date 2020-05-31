import os
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from matplotlib import animation
import time

def anime(path): #顺序播放
    plt.figure()
    for i in range(55):
        plt.subplot(1,1,1)
        num=1965+i
        data = pd.read_csv(path)   #列数不同有问题
        data=data.loc[num-1965]           #0指数据的第一行
        lannum=int((data.count()-1)/2)
        y1=[]
        y1.append(data['percent'])
        for i in range(lannum-1):
            column='percent.'+str(i+1)
            y1.append(data[column])
        xr=[]
        xr.append(data['language'])
        for i in range(lannum-1):
            column='language.'+str(i+1)
            xr.append(data[column])
        plt.bar(xr,y1,label=data['year'])
        plt.xlabel('language')
        plt.ylabel('percent')
        plt.title(str(num)+'\'s popular language')
        plt.xticks(rotation=-45) 
        plt.ion()
        plt.pause(0.5)
        plt.clf()
    return

    

def specific(path,num):   #特定一年的展示
    data = pd.read_csv(path)   #列数不同有问题
    data=data.loc[num-1965]           #0指数据的第一行
    #print(data)
    #print(data.count())
    lannum=int((data.count()-1)/2)
    #print(data)
    #print(data.dtypes)
    y1=[]
    y1.append(data['percent'])
    for i in range(lannum-1):
        column='percent.'+str(i+1)
        y1.append(data[column])
    xr=[]
    xr.append(data['language'])
    for i in range(lannum-1):
        column='language.'+str(i+1)
        xr.append(data[column])
    plt.bar(xr,y1,label=data['year'])
    plt.xlabel('language')
    plt.ylabel('percent')
    plt.title(str(num)+'\'s popular language')
    plt.xticks(rotation=-45) 
    #plt.legend()
    plt.show()
    #plt.pie(x=y1, labels=xr)
    #plt.show()
    return

    
def mostpopular(path,num):   #第num流行的语言
    kth=num
    data = pd.read_csv(path)   #列数不同有问题
    #data=pd.read_csv(path,header=line)
    #data.describe()
    #print(data)
    #data.hist(bins=100,figsize=(15,10))  #bins表示直方图中柱子的数量，figsize是每张图的大小
    #plt.show()
    if kth==1:
        column='language'
    else:
        column='language.'+str(kth-1)
    x=data['year']
    y1=data[column]
    plt.plot(x,y1,label='most popular')
    plt.xlabel('year')
    plt.ylabel('language')
    plt.title(str(kth)+'th popular language')
    #plt.legend()
    filename=str(num)+'th popular language'
    plt.savefig(filename)
    plt.show()
    #my_x_ticks=np.arrange(1960,2020,5)
    #plt.xticks(my_x_ticks)
    #my_yticks=np.arrange(-15000,10000,1000)
    #plt.yticks(my_yticks)
    
    
def txt2csv(path):
    portion = os.path.splitext(path)#将文件名拆成名字和后缀
    print(portion)
    newname = portion[0] + ".csv"
    os.rename(path, newname)#修改
    data = pd.read_csv(newname)   #列数不同有问题
    print(data.dtypes)
    data[['percent','percent.1','percent.2','percent.3','percent.4','percent.5','percent.6',\
          'percent.7','percent.8','percent.9','percent.10']] = \
    data[['percent','percent.1','percent.2','percent.3','percent.4',\
        'percent.5','percent.6','percent.7','percent.8','percent.9','percent.10']].astype(float)
    print(data.dtypes)
    
if __name__ == '__main__':
    path='testf.txt'
    #path=txt2csv(path)
    #anime(path)
    #mostpopular(path,1)
    #specific(path,2019)
    path='testf.csv'
    #anime(path)
    specific(path,1965)
    for i in range(5):
        mostpopular(path,i+1)
	