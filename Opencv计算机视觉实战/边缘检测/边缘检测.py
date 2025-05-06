import cv2
import numpy as np

def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def Canny():
    img=cv2.imread('./img.png')
    v1=cv2.Canny(img,100,200)
    v2=cv2.Canny(img,150,200)
    res=np.hstack((v1,v2))
    cv_show(res,'res')

    
if __name__ == '__main__':
    Canny()