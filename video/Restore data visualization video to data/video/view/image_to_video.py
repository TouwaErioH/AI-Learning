import cv2
import numpy as np
import os


img_array = []
for i in range(1, 6):
    filename = str(i) + 'th popular language.png'
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    for _ in range(20):
        img_array.append(img)


out = cv2.VideoWriter('line chart.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
