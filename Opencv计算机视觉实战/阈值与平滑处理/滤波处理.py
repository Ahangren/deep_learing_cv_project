import cv2
import numpy as np

def threshold():
    img=cv2.imread('./img.png',cv2.IMREAD_GRAYSCALE)

    ret,dst=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    ret,dst1=cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
    cv2.imshow('cat',dst)
    cv2.imshow('ccat1',dst1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def average_filtering():
    img=cv2.imread('./img.png')
    img1=cv2.boxFilter(img,-1,(10,10))
    img2=cv2.GaussianBlur(img,(5,5),0)
    cv2.imshow('cat2',img2)
    cv2.imshow('cat1',img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    average_filtering()