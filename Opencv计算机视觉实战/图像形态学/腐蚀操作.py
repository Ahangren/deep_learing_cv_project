import cv2
import numpy as np


def corrosion_operation():
    img=cv2.imread('./img.png')
    kernel=np.ones((5,5,),np.uint8)
    corrosion=cv2.erode(img,kernel,iterations=5)
    cv2.imshow('corrosion',corrosion)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def expand_operation():
    img=cv2.imread('./img.png')
    kernel=np.ones((5,5,),np.uint8)
    dilate=cv2.dilate(img,kernel,iterations=5)
    cv2.imshow('dilate',dilate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def open_operation():
    img=cv2.imread('./img.png')
    kernel=np.ones((5,5),np.uint8)
    open_new=cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
    cv2.imshow('open_new',open_new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def close_operation():
    img=cv2.imread('./img.png')
    kernel=np.ones((5,5),np.uint8)
    close=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)
    cv2.imshow('close',close)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gradient_operation():
    img=cv2.imread('./img.png')
    kernel=np.ones((5,5,),np.uint8)
    gradient=cv2.morphologyEx(img,cv2.MORPH_GRADIENT,kernel)
    cv2.imshow('gradient',gradient)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def tophat_operation():
    img=cv2.imread('./img.png')
    kernel=np.ones((5,5,),np.uint8)
    tophat=cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel)
    cv2.imshow('tophat',tophat)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

def blackhat_operation():
    img=cv2.imread('./img.png')
    kernel=np.ones((5,5,),np.uint8)
    blackhat=cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)
    cv2.imshow('blackhat',blackhat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    blackhat_operation()