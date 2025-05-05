import cv2
vc=cv2.VideoCapture('video1.mp4')
# 先判断一下
if vc.isOpened():
    open,frame=vc.read()
else:
    open=False

while open:
    ret,frame=vc.read()
    if frame is None:
        break
    if ret==True:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cv2.imshow('result',gray)
        if cv2.waitKey(100) & 0xff==27:
            break
vc.release()
cv2.destroyAllWindows()