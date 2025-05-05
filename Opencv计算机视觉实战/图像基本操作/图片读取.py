import cv2
import matplotlib.pyplot as plt
import numpy as np

def read_img():

    img=cv2.imread('img.png')  # opencv默认读取的是RGB格式
    print(img)  # 打印出来的是三维的图像矩阵
    # 如果我们需要读取灰度图片的话，我们可以采用以下的方式,添加一个参数即可：
    img=cv2.imread('./img_1.png',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('cat',img)
    # 等待，0表示键盘任意键终止，如果为1000代表1000毫秒结束显示
    cv2.waitKey(0)
    cv2.destroyAllWindows() # 关闭所有图片窗口


def care_a_hang():
    img=cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)
    img1=img[100:200,100:400]
    cv2.imshow('cat',img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def Pipeline_separation_consolodation():
    img=cv2.imread('img.png')
    b,g,r=cv2.split(img)
    img1=cv2.merge((b,g,r))
    cv2.imshow('cat',r)
    img_copy=img.copy()
    img_copy[:,:,0]=0
    img_copy[:,:,1]=0
    cv2.imshow('cat1',img_copy)
    cv2.waitKey(0)
# 边界填充
def boundary_fill():
    img=cv2.imread('img.png')
    cv2.imshow('cat',img)

    top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
    replicate=cv2.copyMakeBorder(img,top_size,bottom_size,left_size,right_size,cv2.BORDER_CONSTANT)

    cv2.imshow('cat1',replicate)

    cv2.waitKey(0)

# 对像素点直接操作
def pixel_operation():
    img=cv2.imread('img.png')
    img+=10
    cv2.imshow('cat',img)
    # 改变图片大小
    img=cv2.resize(img,(300,500))
    cv2.imshow('cat',img)

    cv2.waitKey(0)
# 图片融合
def img_fuse():
    img1=cv2.imread('img.png')
    img2=cv2.imread('./img_1.png')
    img1=cv2.resize(img1,(500,500))
    img2 = cv2.resize(img2, (500, 500))
    res=cv2.addWeighted(img1,0.3,img2,0.7,0)
    print(res)
    cv2.imshow('cat1',img2)
    cv2.imshow('cat',res)
    cv2.waitKey(0)
if __name__ == '__main__':
    img_fuse()