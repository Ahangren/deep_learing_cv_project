import pandas as pd
import numpy as np

def create_data():
    # 生成示例数据
    data = {
        "x": [
            "r",
            "g",
            "r",
            "b",
            "g",
            "g",
            "r",
            "r",
            "b",
            "g",
            "g",
            "r",
            "b",
            "b",
            "g",
        ],
        "y": [
            "m",
            "s",
            "l",
            "s",
            "m",
            "s",
            "m",
            "s",
            "m",
            "l",
            "l",
            "s",
            "m",
            "m",
            "l",
        ],
        "labels": [
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "A",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
            "B",
        ],
    }
    data = pd.DataFrame(data, columns=["labels", "x", "y"])
    return data

data=create_data()
print(data)

def get_p_labels(labels):
    labels=list(labels)
    p_label={}
    for label in labels:
        p_label[label]=labels.count(label)/float(len(labels))
    return p_label

p_labels=get_p_labels(data['labels'])
print(p_labels)

train_data=np.array(data.iloc[:,1:])
print(train_data)

labels=data['labels']
label_index=[]
for y in p_labels.keys():
    temp_index=[]

    for i,label in enumerate(labels):
        if label == y:
            temp_index.append(i)
        else:
            pass
    label_index.append(temp_index)
print(label_index)

# 遍历 train_data 中的第一列数据，提取出里面内容为r的数据
x_index = [
    i for i, feature in enumerate(train_data[:, 0]) if feature == "r"
]  # 效果等同于求类别索引中 for 循环

print(x_index)

# 取集合 x_index （x 属性为 r 的数据集合）与集合 label_index[0]（标签为 A 的数据集合）的交集
x_label = set(x_index) & set(label_index[0])
print("既符合 x = r 又是 A 类别的索引值：", x_label)
x_label_count = len(x_label)
# 这里就是用类别 A 中的属性 x 为 r 的数据个数除以类别 A 的总个数
print("先验概率 P(r|A):", x_label_count / float(len(label_index[0])))  # 先验概率的计算公式

def get_P_fea_lab(P_label, features, data):
    # P(\text{特征}∣种类) 先验概率计算
    # 该函数就是求 种类为 P_label 条件下特征为 features 的概率
    P_fea_lab = {}
    train_data = data.iloc[:, 1:]
    train_data = np.array(train_data)
    labels = data["labels"]
    # 遍历所有的标签
    for each_label in P_label.keys():
        # 上面代码的另一种写法，这里就是将标签为 A 和 B 的数据集分开，label_index 中存的是该数据的下标
        label_index = [i for i, label in enumerate(labels) if label == each_label]

        # 遍历该属性下的所有取值
        # 求出每种标签下，该属性取每种值的概率
        for j in range(len(features)):
            # 筛选出该属性下属性值为 features[j] 的数据
            feature_index = [
                i
                for i, feature in enumerate(train_data[:, j])
                if feature == features[j]
            ]

            # set(x_index)&set(y_index) 取交集，得到标签值为 each_label,属性值为 features[j] 的数据集合
            fea_lab_count = len(set(feature_index) & set(label_index))
            key = str(features[j]) + "|" + str(each_label)  # 拼接字符串

            # 计算先验概率
            # 计算 labels 为 each_label下，featurs 为 features[j] 的概率值
            P_fea_lab[key] = fea_lab_count / float(len(label_index))
    return P_fea_lab


features = ["r", "m"]
p_fea_lab=get_P_fea_lab(p_labels, features, data)
print(p_fea_lab)