import imutils
import numpy as np
import cv2


def load_template_digits():
    """加载并处理数字模板"""
    template=cv2.imread('./templates/img_1.png')
    template=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    template=cv2.threshold(template,10,255,cv2.THRESH_BINARY_INV)[1]

    # 查找轮廓并且排序
    cnts=cv2.findContours(template.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts=imutils.grab_contours(cnts)
    cnts=sorted(cnts, key=lambda c:cv2.boundingRect(c)[0])

    digits={}

    for (i,c) in enumerate(cnts):
        (x,y,w,h)=cv2.boundingRect(c)
        roi=template[y:y+h,x:x+w]
        roi=cv2.resize(roi,(57,84))
        digits[i]=roi
    return digits


