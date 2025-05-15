import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt


# -------------------- 数据预处理 --------------------
def preprocess_data():
    # 读取数据并聚类
    data = pd.read_csv("./data/weather.csv", encoding='utf-8')
    data = data.drop(columns=data.columns[-1])

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    # 聚类并保存
    from sklearn.cluster import KMeans
    kms = KMeans(n_clusters=6)
    Y = kms.fit_predict(X_scaled)
    data['class'] = Y
    data.to_csv('./data/weather_new.csv', index=False)
    return data


# -------------------- 数据集加载 --------------------
def load_data(test_size=0.2):
    data = pd.read_csv('./data/weather_new.csv')
    X = data.iloc[:, :-1].values
    y = data['class'].values

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # 转换为 PyTorch 张量
    X_train = torch.FloatTensor(X_train).unsqueeze(1)  # [batch, 1, features]
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test).unsqueeze(1)
    y_test = torch.LongTensor(y_test)

    return X_train, y_train, X_test, y_test


# -------------------- 模型定义 --------------------
class WeatherClassifier(nn.Module):
    def __init__(self, input_features=6, num_classes=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * input_features, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv(x)  # [batch, 128, features]
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


# -------------------- 训练函数 --------------------
def train_model():
    # 加载数据
    X_train, y_train, X_test, y_test = load_data()
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = WeatherClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0
    for epoch in range(100):
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证集评估
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        print(f'Epoch {epoch + 1}, Loss: {train_loss / len(train_loader):.4f}, Acc: {acc:.4f}')

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), './data/best_model.pth')

    print(f'Best Accuracy: {best_acc:.4f}')


# -------------------- 测试函数 --------------------
def test_model():
    model = WeatherClassifier()
    model.load_state_dict(torch.load('./data/best_model.pth'))
    model.eval()

    _, _, X_test, y_test = load_data()
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=50)

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {correct / total:.4f}')


# -------------------- 主程序 --------------------
if __name__ == '__main__':
    preprocess_data()  # 只需运行一次
    train_model()
    test_model()