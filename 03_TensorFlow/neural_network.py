from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import numpy as np

X = np.array([[0., 0.], [1., 1.]])  # 特征数据, 这代表了我们的输入神经元有两个
y = np.array([0, 1])
print(X.shape)
print(y.shape)

# 分类器   solver='sgd' 优化算法 随机梯度下降    alpha=1e-5 学习率
# activation='relu' 激活函数 relu HiddenLayer用  激活函数一般是指隐藏层,
# max_iter=2000 最大迭代次数 , tol=1e-4 忍受度，最近两次loss的差值小于这个数就可以停止迭代。在sklearn中，是连续10次
# 是否要把迭代信息打印出来 verbose=True,
# hidden_layer_sizes=(5, 2) 隐藏层是个二元组，Hidden共2层，第一层五个神经元，第二层2个神经元(隐藏节点)
# 这个地方默认是有截距项的。 shuffle=Ture 样本进来的时候，用不用打乱顺序
clf = MLPClassifier(solver='sgd', alpha=1e-5, activation='relu',
                    hidden_layer_sizes=(5, 2), max_iter=2000, tol=1e-4, verbose=True)

# OK 根据上边的代码，我们的神经网络大概长这样   二分类的话，一个输出节点就够了。  默认全连接，就不画了
#       Input Layer         HiddenLayer_1       HiddenLayer_2       OutPut Layer
#                               O
#           O                   O                    O
#                (2行5列的w)     O     (5行2列的w)                          O
#           O                   O                    O
#                               O

# 进行训练
clf.fit(X, y)

# 训练完成后，拿新的数据去预测
predicted_value = clf.predict([[2, 2], [-1, -2]])
print('predicted_value:', predicted_value)

predicted_proba = clf.predict_proba([[2, 2], [-1, -2]])
print('predicted_proba:', predicted_proba)

print([coef.shape for coef in clf.coefs_])
print('================================================')
print([coef for coef in clf.coefs_])