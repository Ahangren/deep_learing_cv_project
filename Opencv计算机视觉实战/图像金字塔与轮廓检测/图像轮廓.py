import cv2
import numpy as np



def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def binary_image():
    img=cv2.imread('./img.png')
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    contours,hierarchy=cv2.findContours(
        thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(img,contours,-1,(0,0,255),2)
    cv2.imshow('img',img)

    cv_show(thresh,'thresh')

if __name__ == '__main__':
    binary_image()