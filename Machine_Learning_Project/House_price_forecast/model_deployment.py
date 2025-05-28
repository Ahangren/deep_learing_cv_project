from seaborn import load_dataset
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import joblib
from flask import Flask,request,jsonify

def generative_model():
    warnings.filterwarnings('ignore')
    df=load_dataset('titanic')

    print(df.head())
    X = df[["pclass", "sex", "embarked"]]  # 特征
    y = df["alive"]  # 目标

    X=pd.get_dummies(X)
    print(X.head())

    model=RandomForestClassifier()
    score=np.mean(cross_val_score(model,X,y,cv=5))
    print(score)

    model.fit(X,y)
    joblib.dump(model,"./data/titanic.pkl")

%writefile predict.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)


@app.route("/", methods=["POST"])  # 请求方法为 POST
def predict():
    json_ = request.json  # 解析请求数据
    query_df = pd.DataFrame(json_)  # 将 JSON 变为 DataFrame
    columns_onehot = [
        "pclass",
        "sex_female",
        "sex_male",
        "embarked_C",
        "embarked_Q",
        "embarked_S",
    ]  # 独热编码 DataFrame 列名
    query = pd.get_dummies(query_df).reindex(
        columns=columns_onehot, fill_value=0
    )  # 将请求数据 DataFrame 处理成独热编码样式
    clf = joblib.load("titanic.pkl")  # 加载模型
    prediction = clf.predict(query)  # 模型推理
    return jsonify({"prediction": list(prediction)})  # 返回推理结果