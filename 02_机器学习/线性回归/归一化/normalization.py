from sklearn.preprocessing import StandardScaler
import numpy as np

# 归一化
# 我们在训练集里边做了归一化，那么我们对测试集也要用相同的归一化去处理，不能区别对待
# 我们做训练集的时候就已经求出来均值和方差了，我们可以将第一行样本的第一列减去训练集这一列的均值除以训练集的σ1

if __name__ == '__main__':
    temp = np.array([1, 2, 3, 5, 5])    # 1行5列，转换为5行1列
    temp = temp.reshape(-1, 1)  # 转换为5行 1列
    scaler = StandardScaler()
    scaler.fit(temp)            # 计算每一列的均值和标准差做归一化
    scaler.mean_            # 平均差
    scaler.var_             # 标准差的平方  方差
    temp = scaler.transform(temp)
    print(temp)

    data = np.array([1, 2, 3, 5, 800001]).reshape(-1, 1)
    print(data)
    scaler = StandardScaler()
    scaler.fit(data)
    print(scaler.mean_)
    print(scaler.var_)
    # 做完归一化之后，大大的缩小了数据之间的距离
    data = scaler.transform(data)
    print(data)