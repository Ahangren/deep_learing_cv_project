import cv2
import numpy as np

def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


img=cv2.imread('./img.png')
cv_show(img,'img')
print(img.shape)

# 上采样
up=cv2.pyrUp(img)
cv_show(up,'up')
print(up.shape)

# 下采样
down=cv2.pyrDown(img)
cv_show(down,'down')
print(down.shape)

