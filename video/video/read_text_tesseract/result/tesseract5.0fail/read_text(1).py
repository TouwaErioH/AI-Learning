# -*-encoding:utf-8-*-
import pytesseract
from PIL import Image
import os
import time


def test():
    dir_path = './'
    header = 0
    now_year = 1964
    with open('result.txt', 'a+',encoding = 'utf-8') as f:
        for i, image_name in enumerate(sorted(os.listdir(dir_path))):
            # 假如没有表头要将表头写入
            
            if header == 0:
                f.write("year,language,percent,language,percent,language,percent,language,percent,language,percent,language,percent,language,percent,language,percent,language,percent,language,percent,language,percent")
                header = 1
            
            # 语言名和数字之前都要添加逗号
            if 'line' in image_name:
                print(image_name)
                # 假如now_year改变了就要写入年份数据
                if str(now_year) not in image_name:
                    now_year = now_year + 1
                    f.write('\n')
                    f.write(str(now_year))
                f.write(',')
                image = Image.open(image_name)
                if '_0_line' in image_name:
                    text = pytesseract.image_to_string(image, lang='eng',\
                                                       config='--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+#')
                else:
                    text = pytesseract.image_to_string(image, lang='eng',\
                                                   config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789.')
                print(text)
                f.write(str(text))
        f.write('\n\n')


if __name__ == '__main__':
    start_time = time.time()
    test()
    end_time = time.time()
    print(end_time - start_time)
