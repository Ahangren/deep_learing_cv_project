import numpy as np
from torchvision import transforms, datasets
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 数据增强和预处理
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.01,
        contrast=0.01,
        saturation=0.01,
        hue=0.01,
    ),
    transforms.RandomResizedCrop(150, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

root_dir = "./data/PetImages"

# 加载数据集
train_data = datasets.ImageFolder(
    root=os.path.join(root_dir, 'train'),
    transform=transform
)

test_data = datasets.ImageFolder(
    root=os.path.join(root_dir, 'test'),
    transform=transform
)

# 创建DataLoader
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 打印数据集信息
print(f'训练类别数量：{len(train_data.classes)}')
print(f'训练类别名称：{train_data.classes}')
print(f'训练数据集数量：{len(train_data)}')
print(f'测试集数量：{len(test_data)}')


# 可视化函数
def imshow(img, title):
    img = img.numpy().transpose((1, 2, 0))
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')


# 修正后的模型结构
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # 修正输入通道为64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # 修正输入通道为128
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

        # 修正全连接层输入尺寸（根据实际特征图计算）
        self.fc = nn.Sequential(
            nn.Linear(256 * 9 * 9, 512),  # 150x150 -> 75x75 -> 37x37 -> 18x18 -> 9x9
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, len(train_data.classes)),  # 输出类别数动态匹配
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


model = Net().to(device)
lr = 1e-3
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# 参数统计函数
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'模型总参数：{total_params}')
    print(f"模型可训练参数数量：{trainable_params}")


count_parameters(model)


# 修正后的训练函数
def train(dataloader, model, loss_fn, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix(loss=loss.item())

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(dataloader)
    return avg_loss, accuracy


# 修正后的评估函数
def evaluate(dataloader, model, loss_fn):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc='Evaluating', leave=False)
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(dataloader)
    return avg_loss, accuracy


# 训练循环
num_epochs = 20
train_loss, train_acc = [], []
test_loss, test_acc = [], []

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # 训练和评估
    epoch_train_loss, epoch_train_acc = train(train_loader, model, loss_fn, optimizer)
    epoch_test_loss, epoch_test_acc = evaluate(test_loader, model, loss_fn)

    # 更新学习率
    scheduler.step()

    # 记录指标
    train_loss.append(epoch_train_loss)
    train_acc.append(epoch_train_acc)
    test_loss.append(epoch_test_loss)
    test_acc.append(epoch_test_acc)

    # 打印日志
    template = ('Epoch:{:2d}, Train_acc:{:.1f}%, Train_loss:{:.3f}, Test_acc:{:.1f}%, Test_loss:{:.3f}')
    print(template.format(epoch + 1, epoch_train_acc, epoch_train_loss, epoch_test_acc, epoch_test_loss))

# 可视化结果
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), train_loss, label='Train Loss')
plt.plot(range(num_epochs), test_loss, label='Test Loss')
plt.legend()
plt.title("Loss Curve")

plt.subplot(1, 2, 2)
plt.plot(range(num_epochs), train_acc, label='Train Accuracy')
plt.plot(range(num_epochs), test_acc, label='Test Accuracy')
plt.legend()
plt.title("Accuracy Curve")
plt.show()

# 保存模型
save_path = './saved_models/'
os.makedirs(save_path, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_path, 'model_weights.pth'))