# -*-encoding:utf-8-*-
import pytesseract
from PIL import Image
import cv2


def main():
    for i in range(5):
        image = Image.open(str(2000+i) + ' Q4.jpg')
        image = image.convert('L') 
        # 只保留黑色
        size = image.size
        
        imb = Image.new('L', size)
        pixadata = image.load()
        pixdata = imb.load()
        for p in range(size[0]):
            for j in range(size[1]):
                pixdata[p,j] = pixadata[p,j]
                
        w= size[0]
        h = size[1]
        pixdata = image.load()
        print(pixdata)
        print(type(pixdata))
        for x in range(1, w-1):
            for y in range(1, h-1):
                count = 0
                if pixdata[x,y-1] > 245:
                    count = count + 1
                if pixdata[x,y+1] > 245:
                    count = count + 1
                if pixdata[x-1,y] > 245:
                    count = count + 1
                if pixdata[x+1,y] > 245:
                    count = count + 1
                if count > 2:
                    pixdata[x,y] = 255
        savename=str(2000+i) + 'new Q4.jpg'
        imb.save(savename)
        imp=cv2.imread(savename,cv2.IMREAD_GRAYSCALE)
        # 自定义灰度界限，大于这个值为黑色，小于这个值为白色
        ret,test=cv2.threshold(imp,127,255,cv2.THRESH_BINARY)
        cv2.imwrite(str(2000+i)+"test.jpg",test)

        
if __name__ == '__main__':
    main()
