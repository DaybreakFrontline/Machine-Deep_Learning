# 通用模型
# 通用模型可以用来设计非常复杂、任意拓扑结构的神经网络
# 类似于序列模型，通用模型通过函数化的应用接口来定义模型
# 使用函数化的应用接口有众多好处。比如：决定函数执行结果的唯一要素是其返回值
# 返回值的唯一要素则是其参数。这大大的减轻了代码测试的工作量

# 在通用模型中，定义的时候，从输入的多维矩阵开始，然后定义各个层的要素
# 将输入层和输出层作为参数纳入通用模型中就可以定义一个模型对象

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model   # 通用模型

# 定义输入层
input = Input(shape=(784, ))
# 定义各个连接层，假设从输入层开始，定义两个隐藏层，都有64个神经单元，都使用
x = Dense(64, activation='relu')(input)
x = Dense(64, activation='relu')(x)
# 定义输出层，使用最近的隐含层作为参数
y = Dense(10, activation='softmax')(x)

# 所有要素都齐备之后，就可以定义模型对象了，参数很简单，分别是输入和输出
# 中间的各种消息
model = Model(inputs=input, outputs=y)
model.summary()

# 当模型对象定义完之后，就可以进行编译了，并对数据进行拟合，拟合的时候也有两个参数
# 分别对应于输入和输出    optimizer n. 优化器        loss 损失函数       metrics 评估指标  可以传入数组
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# 训练
# model.fit(x_train, y_train)
# 评估  evaluate 评估
# model.evaluate(x_test, y_test)