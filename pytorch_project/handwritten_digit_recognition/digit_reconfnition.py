import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc=nn.Sequential(
            torch.nn.Linear(1600,50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50,10),
        )

    def forward(self,x):
        batch_size=x.size(0)
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(batch_size,-1)
        x=self.fc(x)
        return x

model=Net().to(device)
model_path='./data/model_weights.pth'

model.load_state_dict(torch.load(model_path))
model.eval()


# 预测
def predict_image(image_path,model):
    image=Image.open(image_path)
    transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])
    image=transform(image)
    image=image.to(device)
    image=image.unsequeeze(0)
    with torch.no_grad():
        output=model(image)
        _,predicted=torch.max(output.data,1)

    return predicted.itm()

def show_img(img_path):
    import matplotlib.pyplot as plt
    img=Image.open(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    predicted_digit=predict_image(img_path,model)
    return (f'Predicted digit: {predicted_digit}')

