import numpy as np
import cv2
import sys
import os
import time
# 215,77   66  1776,939    1865,954


def judgeQ4(image):  # 判断帧中内容是否为 Q4。判断标准为取4的-部分所在区域(1776,939    1865,954)，其中白色(>240)的像素点大于400则认为不是，否则认为是
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cnt = 0
    for i in range(1776, 1865):
        for j in range(939, 954):
            if image[j][i] >= 240:
                cnt += 1
    if cnt >= 400:
        return False
    else:
        return True


def judgecover(image):  # 判断是否有覆盖。判断方法为截取高度在70-1080，x坐标在218-219的矩形部分，逐行判断，若该行白色像素点
    # 数量>8则认为该行是白色的。取70开始是为了避免坐标轴上00,10,20会将上方的空白分成2部分
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (h, w) = image.shape  # 然后计算出连续的白色行的长度，即为白色矩形的宽度，也就是数据的柱子之间白色的距离
    h_ = [0]*h  # [0,0,0,0,0,0,...]              #当最大的间距<26时认为柱子无重叠或重叠的很少可以接受
    for y in range(70, 1080):
        cnt = 0
        for x in range(218, 229):
            if image[y, x] >= 240:
                cnt += 1
                if cnt >= 8:
                    h_[y] = 1
                    break
    len = 0
    npp = []
    for i in range(h):
        if h_[i] == 1:
            len += 1
        else:
            if len != 0:
                npp.append(len)
            len = 0
    if npp:  # 可能是空的
        npp = npp[0:-1]
        # print(npp)
        if npp:
            if np.max(npp) >= 26:
                return False
            else:
                return True
        return True
    return True
    return getHProjection(image)


def getframes(path):  # 每隔5帧截一帧。判断是否为Q4以及柱子的覆盖可以接收后保存
    goal_frame = np.array([])
    cap = cv2.VideoCapture(path)  # 返回一个cap对象，从0帧开始
    if(cap.isOpened()):
        print('c')
    num = 0
    i = 0
    gap = 0
    index = []
    # gap_frame = 38  # 每季度持续38帧
    gap_frame = 5  # 隔10帧提取一张图片
    i = 0
    # 假如gold_img_flag为0则说明前一张图片不符合条件
    gold_img_flag = 0
    # 必须要判断失败的图片差别较大的时候才进行保存,这样可以防止在排名发生变化的时候的较短间隔内截取两张图片
    false_gap = 0
    while gap < 8480:
        # print(i)
        gap += gap_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, gap)  # 设置要获取的帧号
        success, frame = cap.read()  # read方法返回一个布尔值和一个视频帧。若帧读取成功，则返回True
        # print(type(frame)) #numpy.ndarray
        # print(frame.shape) #(1080,1920,3)
        if judgeQ4(frame) and judgecover(frame):
            false_gap = 0
            goal_frame = frame
            gold_img_flag = 1
            # cv2.imwrite(str(i) + '.jpg', frame)
            num += 1
            index.append(i)
        elif gold_img_flag == 1 and false_gap > 5:
            false_gap = false_gap + 1
            print(1965+i)
            cv2.imwrite('./judged/'+str(1965+i)+'.jpg', goal_frame)
            i = i+1
            gold_img_flag = 0
        else:
            false_gap = false_gap + 1
    return num, index


def get2019():  # 2019年没有Q4，特殊处理
    gap = 8450
    cap = cv2.VideoCapture(path)  # 返回一个cap对象，从0帧开始
    cap.set(cv2.CAP_PROP_POS_FRAMES, gap)
    success, frame = cap.read()
    cv2.imwrite('./judged/'+str(2019) + '.jpg', frame)


def turn2years(index):  # 上面得到的图片是按帧编号的，接下来取其中的55幅，代表各年。由于同一年的Q4帧间距离都比较近而不同年间距离比较远
    # for i in range()    #故在index中判断，较为接近的index认为是同一年的一组index，取同一组index中最大的一个index的图片作为代表该年的图片
    # 取前后两个index差距大于20 (即帧差距大于100,帧率30000/1001，时间差约3.44秒)
    savepath = './judged/'
    last = -1
    temp = 1965
    for i in range(len(index)):
        if index[i]-last > 20:
            img = cv2.imread(str(last)+'.jpg')
            cv2.imwrite(savepath+str(temp)+'.jpg', img)
            temp += 1
            last = index[i]
        else:
            last = index[i]
    img = cv2.imread('2019.jpg')
    cv2.imwrite(savepath+str(temp)+'.jpg', img)


if __name__ == '__main__':
    start = time.time()
    path = 'video.mp4'
    # savepath='./judged/'+str(i)+'.jpg'
    framenum, index = getframes(path)
    get2019()
    # index.append(2019)
    # print(index)
    # turn2years(index)
    end = time.time()
    print('totally cost', end-start)  # 272.8170356750488  4分半
