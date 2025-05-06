import cv2
import numpy as np

def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gradient_operation():
    img=cv2.imread('img.png')
    sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobelx=cv2.convertScaleAbs(sobelx)
    cv_show(sobelx,'sobelx')

def gradient_operation_y():
    img=cv2.imread('img.png')
    sobelx=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    sobelx=cv2.convertScaleAbs(sobelx)
    cv_show(sobelx,'sobelx')

def gradient_operation_x_y():
    img=cv2.imread('img.png')
    sobelx_x=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobelx_x=cv2.convertScaleAbs(sobelx_x)
    sobelx_y=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    sobelx_y=cv2.convertScaleAbs(sobelx_y)
    sobelxy=cv2.addWeighted(sobelx_x,0.5,sobelx_y,0.5,0)
    cv_show(sobelxy,'sobelxy')



if __name__ == '__main__':
    gradient_operation_x_y()