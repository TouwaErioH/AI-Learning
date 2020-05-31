# -*-encoding:utf-8-*-
import pytesseract
from PIL import Image
from aip import AipOcr

# https://segmentfault.com/a/1190000015144721 general_basic general accurate_basic accurate
# 2017  {'words': 'C #'}, {'words': '8 .75'}, {'words': 'PHP '}, {'words': '8 .18'}, 有一次可能是网络问题？丢掉了8.75
config = {
    'appId': '18058315',
    'apiKey': 'h09BIebxL2CdOx6eGnDZpLb8',
    'secretKey': 'qd6zu3NL2L04f0vToxO3RY5hCi49FtvQ'
}
 
client = AipOcr(**config)
 
def get_file_content(file):
    with open(file, 'rb') as fp:
        return fp.read()
def main():
    header=0
    for k in range(55):
      with open('testf.txt', 'a+') as f:
        #first=1 # 第一个字符不需要写入 ','，否则要写 ', text';  csv以','分隔，有','会认为后面还有数据;还需要表头
        if header==0:
            f.write("year,language,percent,language,percent,language,percent,language,percent,language,percent,language,percent,language,percent,language,percent,language,percent,language,percent,language,percent")
            f.write('\n')
            header=1
        #data=f.read()
        #print(data)
        image_path=str(1965 + k) + '.jpg'
        image = get_file_content(image_path)
        
        cnt=0 #丢失字符个数
        last=-1 #last==0，上一个是数字；==1，上一个是字符
        flag=-1 #一开始是连续的数字，需要跳过;
        options={}
        options["language_type"]="ENG"
        #options["recognize_granularity"]="big"
        result = client.basicAccurate(image,options) #高精度会把数字分开。如2000-> 2 000
        #result = client.basicGeneral(image,options) #有时候会把C++识别为 +t   enhancedGeneral，...
        if result.get('words_result')!=None:                   #可以直接写入CSV，这里先写到TXT
            print(result['words_result'])
            tpstr=str(1965+k)
            yearnum=1965+k
            if yearnum <2000:
                yearnum='1 '+str(yearnum%1000)
            elif yearnum <2010:
                yearnum='2 00'+str(yearnum%2000)
            else:
                yearnum='2 0'+str(yearnum%2000)
            print(yearnum)
            f.write(tpstr)
            for i in result.get('words_result'):
                tpstr=i.get('words')#str
                #print(tpstr+"x")       #跳过最后的年份。如2009 xxx；以及populao languages
                if tpstr.find(yearnum)!=-1 or tpstr.find('Popular')!=-1 or tpstr.find('Programming')!=-1 or \
                tpstr.find('Languages') != -1:
                    continue
                if flag==-1: #跳过开始的尺度 0.00 10.00 20.00 ...
                    if tpstr.find(".00")!=-1:
                        print("???")
                        continue
                    else:
                        flag=1

                if tpstr=='C ' or tpstr=='E ':   #有时候会把单独的字符C认为是E.单个字符C,R有时会丢失;C排第一时可能会掉
                    cnt+=1
                    f.write(",C ")
                    last=1
                elif tpstr=='R ':
                    cnt+=1
                    f.write(","+tpstr)
                    last=1
                elif tpstr.find('.')!=-1 and (last==-1 or last==0):   #tpstr.find('.')!=-1 识别出了数字且上一个也是数字
                    if cnt ==0:
                        f.write(",C ")
                        tpstr=tpstr[0]+tpstr[2:]   #识别结果为 类似 2 1.01，中间有空格
                        f.write(',')
                        f.write(tpstr)
                        cnt+=1
                        last=0
                    else:
                        f.write(",R ")
                        tpstr=tpstr[0]+tpstr[2:]   #识别结果为 类似 2 1.01，中间有空格
                        f.write(',')
                        f.write(tpstr)
                        last=0
                elif tpstr.find('.')!=-1:
                    tpstr=tpstr[0]+tpstr[2:]   #识别结果为 类似 2 1.01，中间有空格.只能写str。后续再csv改为float
                    f.write(',')
                    f.write(tpstr)
                    last=0
                else:
                    f.write(','+tpstr)
                    last=1
        print("finish\n\n")
        f.write("\n")


if __name__ == '__main__':
    start=time.time()
    main()
    end=time.time()
    print('totally cost',end-start) #272.8170356750488  4分半
