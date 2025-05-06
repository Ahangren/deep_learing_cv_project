import numpy as np
import cv2
import matplotlib.pyplot as plt

def cv_show(img,name):
    cv2.imshow(name,img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def hist_img():
    img=cv2.imread('./img.png')
    cv_show(img,'img')
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    hist=cv2.calcHist([img],[0],None,[256],[0,256])
    plt.hist(img.ravel(),256,)
    plt.show()

def mask_img():
    img=cv2.imread('./img.png')
    print(img.shape)
    mask=np.zeros(img.shape[:2],np.uint8)
    print(mask.shape)
    mask[100:300,100:400]=255
    masked_img=cv2.bitwise_and(img,img,mask=mask)
    cv_show(masked_img,'mask')

    hist_full=cv2.calcHist([img],[0],None,[256],[0,256])
    hisk_mask=cv2.calcHist([img],[0],mask,[256],[0,256])

    plt.subplot(221), plt.imshow(img, 'gray')
    plt.subplot(222), plt.imshow(mask, 'gray')
    plt.subplot(223), plt.imshow(masked_img, 'gray')
    plt.subplot(224), plt.plot(hist_full), plt.plot(hisk_mask)
    plt.xlim([0, 256])  # 设置x轴的数值显示范围。
    plt.show()

def hist_equilibrium():
    img=cv2.imread('./img.png',0)

    equ = cv2.equalizeHist(img)
    plt.subplot(121)
    plt.hist(img.ravel(), 256)  # 均衡化前

    plt.subplot(122)
    plt.hist(equ.ravel(), 256)  # 均衡化后

    plt.show()

def Fourier_transform():
    img=cv2.imread('./img.png',0)
    rows,clos=img.shape
    nrows=cv2.getOptimalDFTSize(rows)
    ncols=cv2.getOptimalDFTSize(clos)

    padded=cv2.copyMakeBorder(img,0,nrows-rows,0,ncols,cv2.BORDER_CONSTANT,value=0)

    dft=cv2.dft(np.float32(padded),flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift=np.fft.fftshift(dft)

    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    # 6. 显示结果
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == '__main__':

    Fourier_transform()