from PIL import Image
from PIL import ImageEnhance
import cv2
import os
import numpy as np


# 亮度增强
def brightness_ImageEnhance(img):
    enh_bri = ImageEnhance.Brightness(img)
    brightness = 2
    image_brightened = enh_bri.enhance(brightness)
    # image_brightened.show()
    return image_brightened


# 色度增强
def color_ImageEnhance(img):
    enh_col = ImageEnhance.Color(img)
    color = 1.5
    image_colored = enh_col.enhance(color)
    # image_colored.show()
    return image_colored


# 对比度增强
def contrast_ImageEnhance(img):
    enh_con = ImageEnhance.Contrast(img)
    contrast = 5
    image_contrasted = enh_con.enhance(contrast)
    # image_contrasted.show()
    return image_contrasted


# 锐度增强
def sharpness_ImageEnhance(img):
    enh_sha = ImageEnhance.Sharpness(img)
    sharpness = 3.0
    image_sharped = enh_sha.enhance(sharpness)
    # image_sharped.show()
    return image_sharped


def image_enhance():
    for i in range(55):
        image_path = str(1965 + i) + ' Q4.jpg'
        image = Image.open(image_path)
        image_color = color_ImageEnhance(image)
       # image_sharpness = sharpness_ImageEnhance(image_color)
       # image_contrast = contrast_ImageEnhance(image_color)
        cv2.imwrite(image_path, np.array(image_color))

if __name__ == '__main__':
    image_enhance()