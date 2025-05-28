from typing import final

import numpy as np
import pandas as pd
import math

def create_data():
    # 生成示例数据
    data_value = np.array(
        [
            ["long", "thick", 175, "no", "man"],
            ["short", "medium", 168, "no", "man"],
            ["short", "thin", 178, "yes", "man"],
            ["short", "thick", 172, "no", "man"],
            ["long", "medium", 163, "no", "man"],
            ["short", "thick", 180, "no", "man"],
            ["long", "thick", 173, "yes", "man"],
            ["short", "thin", 174, "no", "man"],
            ["long", "thin", 164, "yes", "woman"],
            ["long", "medium", 158, "yes", "woman"],
            ["long", "thick", 161, "yes", "woman"],
            ["short", "thin", 166, "yes", "woman"],
            ["long", "thin", 158, "no", "woman"],
            ["short", "medium", 163, "no", "woman"],
            ["long", "thick", 161, "yes", "woman"],
            ["long", "thin", 164, "no", "woman"],
            ["short", "medium", 172, "yes", "woman"],
        ]
    )
    columns = np.array(["hair", "voice", "height", "ear_stud", "labels"])
    data = pd.DataFrame(data_value.reshape(17, 5), columns=columns)
    return data

data=create_data()
print(data)

def get_ent(data):
    num_sample=len(data)
    label_counts={}
    for i in range(num_sample):
        each_data=data.iloc[i,:]
        current_label=each_data['labels']
        if current_label not in label_counts:
            label_counts[current_label]=0
        label_counts[current_label]+=1

    ent=0.0
    for key in label_counts:
        pro=float(label_counts[key])/num_sample
        ent-=pro*math.log(pro,2)
    return ent

base_ent=get_ent(data)
print(base_ent)


def get_gain(data,base_ent,feature):
    feature_list=data[feature]
    unique_value=set(feature_list)
    feature_ent=0.0
    for each_feature in unique_value:
        temp_data=data[data[feature]==each_feature]
        weight=len(temp_data)/len(feature_list)
        temp_ent=weight*get_ent(temp_data)
        feature_ent+=temp_ent
    gain=base_ent-feature_ent
    return gain

gain=get_gain(data,base_ent,'hair')

print(gain)

def get_splitpoint(data,base_ent,feature):
    continues_value=data[feature].sort_values().astype(np.float64)
    continues_value=[i for i in continues_value]
    t_set=[]
    t_ent={}
    for i in range(len(continues_value)-1):
        temp_t=continues_value[i]+continues_value[i+1]
        t_set.append(temp_t/2)

    for each_t in t_set:
        temp1_data=data[data[feature].astype(np.float64)>each_t]
        temp2_data=data[data[feature].astype(np.float64)<each_t]
        weight1=len(temp1_data)/len(data)
        weight2=len(temp2_data)/len(data)

        t=weight1*get_ent(temp1_data)+weight2*get_ent(temp2_data)
        t_ent[each_t]=base_ent-t

    print('t_ent:' ,t_ent)
    final_t=max(t_ent,key=t_ent.get)
    return final_t

final_t=get_splitpoint(data,base_ent,'height')
print(final_t)

def choice_1(x,t):
    if x>t:
        return f'>{t}'
    else:
        return f'<{t}'
deal_data=data.copy()
deal_data['height']=pd.Series(
    map(lambda x:choice_1(int(x),final_t),deal_data['height'])
)

print(deal_data)

def choose_feature(data):
    num_features=len(data.columns)-1
    base_ent=get_ent(data)
    best_gain=0.0
    best_feature=data.columns[0]
    for i in range(num_features):
        temp_gain=get_gain(data,base_ent,data.columns[i])
        if temp_gain>best_gain:
            best_gain=temp_gain
            best_feature=data.columns[i]
    return best_feature

def create_tree(data):
    """
    参数:
    data -- 数据集

    返回:
    tree -- 以字典的形式返回决策树
    """
    # 构建决策树
    feature_list = data.columns[:-1].tolist()
    label_list = data.iloc[:, -1]
    if len(data["labels"].value_counts()) == 1:
        leaf_node = data["labels"].mode().values
        return leaf_node  # 第一个递归结束条件：所有的类标签完全相同
    if len(feature_list) == 1:
        leaf_node = data["labels"].mode().values
        return leaf_node  # 第二个递归结束条件：用完了所有特征
    best_feature = choose_feature(data)  # 最优划分特征
    tree = {best_feature: {}}
    feat_values = data[best_feature]
    unique_value = set(feat_values)
    for value in unique_value:
        temp_data = data[data[best_feature] == value]
        temp_data = temp_data.drop([best_feature], axis=1)
        tree[best_feature][value] = create_tree(temp_data)
    return tree

tree=create_tree(deal_data)
print(tree)

def classify(tree, test):
    """
    参数:
    data -- 数据集
    test -- 需要测试的数据

    返回:
    class_label -- 分类结果
    """
    # 决策分类
    first_feature = list(tree.keys())[0]  # 获取根节点
    feature_dict = tree[first_feature]  # 根节点下的树
    labels = test.columns.tolist()
    value = test[first_feature][0]
    for key in feature_dict.keys():
        if value == key:
            if type(feature_dict[key]).__name__ == "dict":  # 判断该节点是否为叶节点
                class_label = classify(feature_dict[key], test)  # 采用递归直到遍历到叶节点
            else:
                class_label = feature_dict[key]
    return class_label

test = pd.DataFrame(
    {"hair": ["long"], "voice": ["thin"], "height": [163], "ear_stud": ["yes"]}
)
print(test)

test["height"] = pd.Series(map(lambda x: choice_1(int(x), final_t), test["height"]))
print(test)
classify(tree, test)