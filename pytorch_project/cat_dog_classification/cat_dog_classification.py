import torch
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets,transforms

device=torch.device('cuda' if torch.cuda.is_available() else 'cup')

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2),
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2)
        )
        self.fc=nn.Sequential(
            nn.Linear(256*9*9,512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2),
        )
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x

model=Net().to(device)
model_path="./data/model_weights.pth"
model.load_state_dict(torch.load(model_path))

model.eval()

def predict_image(image_path,model,classes=['cat','dog']):
    image=Image.open(image_path)
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.01,
            contrast=0.01,
            saturation=0.01,
            hue=0.01
        ),
        transforms.RandomResizedCrop(150,scale=(0.8,1.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        ),

    ])
    image=transform(image)
    image=image.to(device)
    image=image.unsqueeze(0)

    with torch.no_grad():
        output=model(image)
        _,predict=torch.max(output,dim=1)
    return classes[predict.item()]

cat_path="./data/cat.jpg"
dog_path='./data/dog.jpg'

plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
cat_predicted=predict_image(cat_path,model)
plt.imshow(Image.open(cat_path))
plt.title(f"Predicted category: {cat_predicted}")
plt.axis('off')

plt.subplot(1,2,2)
dog_predicted=predict_image(dog_path,model)
plt.imshow(Image.open(dog_path))
plt.title(f'Predicted dog: {dog_predicted}')
plt.axis('off')
plt.show()
