import cv2
import numpy as np
import matplotlib.pyplot as plt


image=cv2.imread('./data/5678.png',cv2.IMREAD_GRAYSCALE)

_,binary_image=cv2.threshold(image,127,255,cv2.THRESH_BINARY_INV)
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(image,cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1,2,2)
plt.imshow(binary_image,cmap='gray')
plt.title('Binary Image')
plt.axis('off')

plt.tight_layout()
plt.show()

contours,_=cv2.findContours(binary_image,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

def sort_contours(contours):
    contours_list=list(contours)

    contours_list.sort(key=lambda c:cv2.boundingRect(c)[0])
    return contours_list

contours=sort_contours(contours)

digit_images=[]

for contour in contours:
    x,y,w,h=cv2.boundingRect(contour)
    if h>20 and w>10:
        padding=30
        digit=binary_image[max(y-padding,0):y+h+padding,max(x-padding,0):x+w+padding]
        digit_images.append(digit)
print(len(digit_images))

plt.figure()
for i in range(len(digit_images)):
    plt.subplot(1, len(digit_images), i + 1)
    plt.tight_layout()
    plt.imshow(digit_images[i], cmap='gray', interpolation='none')
    plt.xticks([])
    plt.yticks([])
plt.show()

# 数字识别
import torch
from PIL import Image
import torchvision.transforms as transforms

device=torch.device('cpu')

class Net(torch.nn.Module):
    def __init__(self):
        # （batch,1,28,28）
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3), #（batch,32,26,26） 输入通道数1输出通道数32 32为小型任务的经验性选择，一般每层增加一倍欠拟合就加过拟合减
            torch.nn.BatchNorm2d(32), # 对卷积层的输出进行批量归一化，使得每个特征图的分布更加稳定，从而加速训练并提高模型性能。
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2), #（batch,32,13,13）
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3), #（batch,64,11,11）
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2), #（batch,64,5,5）
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1600, 50), # 1600 == 64*5*5
            torch.nn.ReLU(),  # 添加ReLU激活函数 增加模型的非线性能力
            torch.nn.Dropout(0.5), # 有效防止过拟合-丢弃率0.5          BN层和dropout层一起用效果不好（ 深层可能不好BN在后Dropout在前也不好
            torch.nn.Linear(50, 10)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层
        x = self.conv2(x)  # 再来一次
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入
        x = self.fc(x)
        return x  # 最后输出的是维度为10的，也就是（对应数学符号的0~9）


model = Net().to(device)

model_path = './data/model_weights.pth'

# 加载模型参数
model.load_state_dict(torch.load(model_path))

# 将模型设置为评估模式
model.eval()


# 预测函数
def predict_image(image, model):
    # image = Image.open(image_path)
    image = Image.fromarray(image)
    # 图像预处理
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转换为灰度
        transforms.Resize((28, 28)),  # 调整到 28x28
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize((0.1307,), (0.3081,))  # 归一化到 [-1, 1]
    ])
    image = transform(image)
    image = image.to(device)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    return str(predicted.item())


#展示图片
import matplotlib.pyplot as plt
img = Image.open('./data/5678.png')
# 显示图像
plt.imshow(img)
plt.axis('off')  # 可选，关闭坐标轴
plt.show()

predict_digit = []

for image in digit_images:
    predict_digit.append(predict_image(image, model))
print(''.join(predict_digit))


















