from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor,  Ridge, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.externals import joblib
import pandas as pd
import numpy as np



def mylinear():
    """
    线性回归直接预测房子价格
    :return: None
    """
    # 获取数据
    lb = load_boston()

    # 分割数据集到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)

    #print(y_train.reshape(-1,1), y_test)

    # 进行标准化处理(?) 目标值处理？
    # 特征值和目标值是都必须进行标准化处理, 实例化两个标准化API
    std_x = StandardScaler()

    x_train = std_x.fit_transform(x_train.reshape(-1,1))
    x_test = std_x.transform(x_test.reshape(-1,1))

    # print(x_train)
    # print(x_test)

    # 目标值
    std_y = StandardScaler()

    y_train = std_y.fit_transform(y_train.reshape(-1,1))
    y_test = std_y.transform(y_test.reshape(-1,1))

    print("-------------------->")
    print(y_train)
    print(y_test)
    print("*********************************")
    print(x_train.reshape(1,-1))
    print(y_train.reshape(1,-1))
    #estimator预测
    #正规方程求解方式预测结果
    lr = LinearRegression()
    #
    lr.fit(x_train.reshape(1,-1),y_train.reshape(1,-1))
    #
    print(lr.coef_)

    # 预测房价结果
    #model = joblib.load("./tmp/test.pkl")

    #y_predict = std_y.inverse_transform(model.predict(x_test))

    #print("保存的模型预测的结果：", y_predict)



    return None




if __name__ == "__main__":
    mylinear()