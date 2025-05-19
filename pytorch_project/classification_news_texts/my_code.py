import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn import metrics
import os
from tqdm import tqdm

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 文件路径
trainingSet_path = "./data/train.txt"
valSet_path = "./data/val.txt"
model_save_path = "CNN_model_pytorch.pth"
testingSet_path = "./data/test.txt"

# 超参数
num_classes = 10  # 类别数
vocab_size = 5000  # 语料词大小
seq_length = 600  # 词长度
embedding_dim = 256  # 词嵌入维度
conv1_num_filters = 128  # 第一层卷积核数量
conv1_kernel_size = 1  # 第一层卷积核大小
conv2_num_filters = 64  # 第二层卷积核数量
conv2_kernel_size = 1  # 第二层卷积核大小
hidden_dim = 128  # 隐藏层维度
dropout_rate = 0.5  # dropout率
batch_size = 64  # 批量大小
learning_rate = 1e-3  # 学习率
patience = 3  # 早停耐心值
min_lr = 1e-5  # 最小学习率


# 自定义数据集类
class TextDataset(Dataset):
    def __init__(self, txt_path, max_length):
        self.preprocessor = TextPreprocessor()
        self.x, self.y = self.preprocessor.word2idx(txt_path, max_length)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.LongTensor(self.x[idx]), torch.FloatTensor(self.y[idx])


# 文本预处理类
class TextPreprocessor:
    def __init__(self):
        self.categories = ["体育", "财经", "房产", "家居", "教育", "科技", "时尚", "时政", "游戏", "娱乐"]
        self.cates_dict = {c: i for i, c in enumerate(self.categories)}

    def read_txt(self, txt_path):
        with open(txt_path, "r", encoding='utf-8') as f:
            data = f.readlines()
        labels = []
        contents = []
        for line in data:
            content,label = line.strip().split('\t')
            labels.append(label)
            contents.append(content)
        return labels, contents

    def get_vocab_id(self):
        vocab_path = "./data/vocab.txt"
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocabs = [word.strip() for word in f.readlines()]
        vocabs_dict = {word: i for i, word in enumerate(vocabs)}
        return vocabs, vocabs_dict

    def get_category_id(self):
        return self.cates_dict

    def word2idx(self, txt_path, max_length):
        vocabs, vocabs_dict = self.get_vocab_id()
        cates_dict = self.get_category_id()

        labels, contents = self.read_txt(txt_path)
        labels_idx = []
        contents_idx = []

        for idx in range(len(contents)):
            tmp = []
            labels_idx.append(cates_dict[labels[idx]])
            for word in contents[idx]:
                if word in vocabs_dict:
                    tmp.append(vocabs_dict[word])
                else:
                    tmp.append(5000)  # 未知词索引
            contents_idx.append(tmp)

        # 转换为numpy数组并填充
        x_pad = torch.zeros((len(contents_idx), max_length), dtype=torch.long)
        for i, seq in enumerate(contents_idx):
            seq_length = min(len(seq), max_length)
            x_pad[i, :seq_length] = torch.LongTensor(seq[:seq_length])

        # 转换为one-hot编码
        y_pad = torch.zeros((len(labels_idx), num_classes), dtype=torch.float)
        for i, label in enumerate(labels_idx):
            y_pad[i, label] = 1.0

        return x_pad.numpy(), y_pad.numpy()

    def word2idx_for_sample(self, sentence, max_length):
        vocabs, vocabs_dict = self.get_vocab_id()
        result = []
        for word in sentence:
            if word in vocabs_dict:
                result.append(vocabs_dict[word])
            else:
                result.append(5000)

        x_pad = torch.zeros((1, max_length), dtype=torch.long)
        seq_length = min(len(result), max_length)
        x_pad[0, :seq_length] = torch.LongTensor(result[:seq_length])
        return x_pad.unsqueeze(0)  # 添加batch维度


# TextCNN模型
class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, conv1_num_filters, conv1_kernel_size, padding='same')
        self.conv2 = nn.Conv1d(conv1_num_filters, conv2_num_filters, conv2_kernel_size, padding='same')
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(conv2_num_filters, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        x = x.permute(0, 2, 1)  # [batch_size, embedding_dim, seq_len] for Conv1d

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.global_pool(x).squeeze(-1)  # [batch_size, conv2_num_filters]

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


# 训练函数
def train_model(model, train_loader, val_loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=min_lr, verbose=True)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.argmax(dim=1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.argmax(dim=1)).sum().item()

            progress_bar.set_postfix({
                'loss': train_loss / (total / batch_size),
                'acc': 100. * correct / total
            })

        # 验证阶段
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f'Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')

        # 学习率调度
        scheduler.step(val_acc)

        # 早停和模型保存
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved with validation accuracy: {val_acc:.2f}%')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping after {patience} epochs without improvement')
                break


# 评估函数
def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels.argmax(dim=1))

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.argmax(dim=1)).sum().item()

    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


# 测试函数
def test_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = outputs.max(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.argmax(dim=1).cpu().numpy())

    print(metrics.classification_report(all_labels, all_preds, target_names=preprocessor.categories))
    return all_preds, all_labels


# 预测单条样本
def predict_sample(model, sentence):
    model.eval()
    preprocessor = TextPreprocessor()
    x_test = preprocessor.word2idx_for_sample(sentence, seq_length).to(device)

    with torch.no_grad():
        output = model(x_test)
        _, pred = output.max(1)

    return preprocessor.categories[pred.item()]


if __name__ == '__main__':
    # 初始化预处理类
    preprocessor = TextPreprocessor()

    # 创建数据集和数据加载器
    train_dataset = TextDataset(trainingSet_path, seq_length)
    val_dataset = TextDataset(valSet_path, seq_length)
    test_dataset = TextDataset(testingSet_path, seq_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = TextCNN().to(device)
    print(model)

    # 训练模型
    train_model(model, train_loader, val_loader, epochs=20)

    # 加载最佳模型
    model.load_state_dict(torch.load(model_save_path))
    print("----- Best model loaded -----")

    # 测试模型
    test_model(model, test_loader)

    # 测试单条样本
    test_text = '5月6日，上海莘庄基地田径特许赛在第二体育运动学校鸣枪开赛。男子110米栏决赛，19岁崇明小囡秦伟搏以13.35秒的成绩夺冠，创造本赛季亚洲最佳。谢文骏迎来赛季首秀，以13.38秒获得亚军'
    prediction = predict_sample(model, test_text)
    print(f'该新闻为: {prediction}')