import torch
import torch.nn as nn
from torchvision import transforms ,datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['figure.dpi']=100

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据集下载
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081,))
])
train_dataset=datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_dataset=datasets.MNIST(root='./data',train=False,download=True,transform=transform)

batch_size=32
train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

# 展示数据集
def show_datas():
    fig=plt.figure()
    for i in range(12):
        plt.subplot(3,4,i+1)
        plt.tight_layout()
        plt.imshow(train_dataset.data[i],cmap='gray',interpolation='none')
        plt.title(f'Labels: {train_dataset.train_labels[i]}')
        plt.xticks([])
        plt.yticks([])
    plt.show()

# 构建简单的cnn网路
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,32,kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc=nn.Sequential(
            nn.Linear(1600,50),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(50,10)
        )

    def forward(self,x):
        batch_size=x.size(0)
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(batch_size,-1)
        x=self.fc(x)
        return x

model=Net().to(device)

def count_parameters(model):
    total_params=sum(p.numel() for p in model.parameters())
    trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'模型总参数数量: {total_params:,}')
    print(f'模型可训练参数数量：{trainable_params:,}')

print(model)
count_parameters(model)

loss_fn=nn.CrossEntropyLoss()
learn_rate=0.01
optimizer=torch.optim.SGD(model.parameters(),lr=learn_rate,momentum=0.9)

def train(data_loader,model,loss_fn,optimizer):
    size=len(data_loader.dataset)
    num_batches=len(data_loader)
    train_loss=0
    train_acc=0.0

    for X,y in data_loader:
        X,y=X.to(device),y.to(device)
        pred=model(X)
        loss=loss_fn(pred,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc+=(pred.argmax(1)==y).type(torch.float).sum().item()
        train_loss+=loss.item()

    train_acc/=size
    train_loss/=num_batches

    return train_acc,train_loss

def test(data_loder,model,loss_fn):
    size=len(data_loder.dataset)
    num_batches=len(data_loder)

    test_loss=0
    test_acc=0.0
    with torch.no_grad():
        for imgs,target in data_loder:
            imgs,target=imgs.to(device),target.to(device)

            target_pred=model(imgs)
            loss=loss_fn(target_pred,target)

            test_loss+=loss.item()
            test_acc+=(target_pred.argmax(1)==target).type(torch.float).sum().item()

        test_acc/=size
        test_loss/=num_batches
        return test_acc,test_loss


epochs=10
train_loss=[]
train_acc=[]
test_loss=[]
test_acc=[]

for epoch in range(epochs):
    model.train()
    epoch_train_acc,epoch_train_loss=train(train_loader,model,loss_fn,optimizer)

    model.eval()
    epoch_test_acc,epoch_test_loss=test(test_loader,model,loss_fn)

    train_acc.append(epoch_train_acc)
    train_loss.append(epoch_train_loss)
    test_acc.append(epoch_test_acc)
    test_loss.append(epoch_test_loss)

    template=('Epoch:{:2d}, Train_acc:{:.1f}%, Train_loss: {:.3f}, Test_acc:{:.1f}%, Test_loss:{:.3f}')
    print(template.format(epoch+1, epoch_train_acc*100, epoch_train_loss, epoch_test_acc*100, epoch_test_loss))


epochs_range=range(epochs)

plt.figure(figsize=(12,3))
plt.subplot(1,2,1)

plt.plot(epochs_range,train_acc,label="Training Accuracy")
plt.plot(epochs_range,test_acc,label='Test Accuracy')

plt.legend(loc='lower right')
plt.title('Training and Test Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range,train_loss,label='Training Loss')
plt.plot(epochs_range,test_loss,label='Test Loss')
plt.legend(loc='upper right')
plt.title('Training and Test Loss')
plt.show()


save_dir='./data'

import os
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

torch.save(model.state_dict(),os.path.join(save_dir,'model_weights.pth'))
















if __name__ == '__main__':
    show_datas()