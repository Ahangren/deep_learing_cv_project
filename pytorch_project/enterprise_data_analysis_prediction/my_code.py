import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# 1. 灰色预测函数 (GM11) 保持不变
def GM11(x0):
    """灰色预测函数实现"""
    x1 = x0.cumsum()  # 1-AGO序列
    z1 = (x1[:len(x1) - 1] + x1[1:]) / 2.0  # 紧邻均值生成序列
    z1 = z1.reshape((len(z1), 1))

    B = np.append(-z1, np.ones_like(z1), axis=1)  # 构造矩阵B
    Yn = x0[1:].reshape((len(x0) - 1, 1))  # 构造矩阵Yn

    # 计算参数 [a, b]^T = (B^T * B)^-1 * B^T * Yn
    [[a], [b]] = np.dot(np.dot(np.linalg.inv(np.dot(B.T, B)), B.T), Yn)

    # 定义灰色预测函数
    f = lambda k: (x0[0] - b / a) * np.exp(-a * (k - 1)) - (x0[0] - b / a) * np.exp(-a * (k - 2))

    # 计算后验差比值和小残差概率
    delta = np.abs(x0 - np.array([f(i) for i in range(1, len(x0) + 1)]))
    C = delta.std() / x0.std()
    P = 1.0 * (np.abs(delta - delta.mean()) < 0.6745 * x0.std()).sum() / len(x0)

    return f, a, b, x0[0], C, P


# 2. 自定义PyTorch数据集类
class TimeSeriesDataset(Dataset):
    """时间序列数据集类"""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 3. 定义神经网络模型
class RegressionModel(nn.Module):
    """回归预测模型"""

    def __init__(self, input_size):
        super(RegressionModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 12),
            nn.ReLU(),
            nn.Linear(12, 24),
            nn.ReLU(),
            nn.Linear(24, 1)
        )

    def forward(self, x):
        return self.net(x)


# 4. 主程序
if __name__ == "__main__":
    # ========== 数据加载与预处理 ==========
    data = pd.read_csv('./data/data.csv', index_col=0)  # 读取数据
    data.index = range(2000, 2020)  # 设置索引为年份

    # 添加未来三年的空行
    for year in [2020, 2021, 2022]:
        data.loc[year] = None

    # 需要进行灰色预测的特征列
    features_for_gm = ['x3', 'x5', 'x7']

    # 对选定特征进行灰色预测
    for col in features_for_gm:
        # 使用2000-2019年数据训练灰色模型
        f, _, _, _, C, _ = GM11(data[col].loc[range(2000, 2020)].values)
        print(f"{col}后验差比值：{C:.4f}")

        # 预测2020-2022年的值
        data[col].loc[2020] = f(len(data) - 2)
        data[col].loc[2021] = f(len(data) - 1)
        data[col].loc[2022] = f(len(data))

        # 保留两位小数
        data[col] = data[col].round(2)

    # 保存灰色预测结果
    data[features_for_gm + ['y']].to_csv('./data/GM11.csv')

    # ========== 神经网络建模 ==========
    # 读取预处理后的数据
    data = pd.read_csv('./data/GM11.csv', index_col=0)

    # 定义特征和目标列
    feature_cols = ['x3', 'x5', 'x7']
    target_col = 'y'

    # 划分训练集(2000-2019)和预测集(2000-2022)
    train_data = data.loc[range(2000, 2020)]
    full_data = data.copy()

    # 数据标准化
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_data[feature_cols])
    train_y = train_data[target_col].values.reshape(-1, 1)

    # 创建数据集和数据加载器
    train_dataset = TimeSeriesDataset(train_X, train_y)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 初始化模型
    model = RegressionModel(input_size=len(feature_cols))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    print("\n开始训练神经网络...")
    for epoch in range(1000):  # 减少epoch次数，使用早停法优化
        model.train()
        total_loss = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 每100个epoch打印一次损失
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    # ========== 预测与结果可视化 ==========
    # 对所有数据进行标准化处理
    full_X = scaler.transform(full_data[feature_cols])

    # 使用模型进行预测
    model.eval()
    with torch.no_grad():
        full_data['y_pred'] = model(torch.FloatTensor(full_X)).numpy() * train_data[target_col].std() + train_data[
            target_col].mean()

    # 保存预测结果
    full_data.to_csv('./data/result.csv')

    # 可视化结果
    plt.figure(figsize=(15, 5))
    plt.plot(full_data.index, full_data['y'], 'b-o', label='Actual')
    plt.plot(full_data.index, full_data['y_pred'], 'r-*', label='Predicted')
    plt.xticks(full_data.index)
    plt.xlabel("Year")
    plt.ylabel("Value")
    plt.title("Actual vs Predicted Values")
    plt.legend()
    plt.grid(True)
    plt.show()