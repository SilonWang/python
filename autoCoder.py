# 去噪自编码器

#导入MNIST数据集

import numpy as np

import sklearn.preprocessing as prep

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

#这里使用一种参数初始化方法xavier initialization，需要对此做好定义工作。

#Xaiver初始化器的作用就是让权重大小正好合适。

#这里实现的是标准均匀分布的Xaiver初始化器。期望值=0，方差=2/(fan_in + fan_out) 

def xavier_init(fan_in, fan_out, constant=1):

    """
    
    目的是合理初始化权重。
    
    参数：
    
    fan_in --行数；
    
    fan_out -- 列数；
    
    constant --常数权重，条件初始化范围的倍数。
    
    return 初始化后的权重tensor.
    
    """

    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))

    high = constant * np.sqrt(6.0 / (fan_in + fan_out))

    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high, dtype=tf.float32)

#定义一个去噪的自编码类

class AdditiveGaussianNoiseAutoencoder(object):

    """
    
    __init__() :构建函数；
    
    n_input : 输入变量数；
    
    n_hidden : 隐含层节点数；
    
    transfer_function: 隐含层激活函数，默认是softplus；
    
    optimizer : 优化器，默认是Adam;
    
    scale : 高斯噪声系数，默认是0.1；
    
    """

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(), scale=0.1):

        self.n_input = n_input

        self.n_hidden = n_hidden

        self.transfer = transfer_function

        self.scale = tf.placeholder(tf.float32)

        self.training_scale = scale

        network_weights = self._initialize_weights()

        self.weights = network_weights

        # 定义网络结构，为输入x创建一个维度为n_input的placeholder，然后

        #建立一个能提取特征的隐含层。(hidden = W1(X + scale) + B1)。 transfer对结果进行激活函数处理

        #建立用于数据复原、重建操作的reconstruction层。(reconstruction = W2 * hidden + B2)

        self.x = tf.placeholder(tf.float32, [None, self.n_input])

        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),
                                                    self.weights['w1']),
                                                    self.weights['b1']))

        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        #首先，定义自编码器的损失函数，在此直接使用平方误差(SquaredError)作为cost。

        #然后，定义训练操作作为优化器self.optimizer对损失self.cost进行优化。 

        #最后，创建Session，并初始化自编码器全部模型参数。

        self.cost = 0.5 *  tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))

        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()

        self.sess = tf.Session()

        self.sess.run(init)
    #初始化weight参数
    #w1取xavier方法初始化数据，其他初始化0
    def _initialize_weights(self):

        all_weights = dict()

        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))

        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))

        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))

        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))

        return all_weights

    #取当前代的损失函数值cost
    #执行过程opt值？

    def partial_fit(self, X):

        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.scale: self.training_scale})

        return cost

    def calc_total_cost(self, X):

        return self.sess.run(self.cost, feed_dict={self.x: X, self.scale: self.training_scale})

    #定义一个transform函数，以便返回自编码器隐含层的输出结果，目的是提供一个接口来获取抽象后的特征。

    def transform(self, X):

        return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})

    def generate(self, hidden=None):

        if hidden is None:

            hidden = np.random.normal(size=self.weights["b1"])

        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):

        return self.sess.run(self.reconstruction, feed_dict={self.x: X,
                                                            self.scale: self.training_scale})

    def getWeights(self):  # 获取隐含层的权重w1.

        return self.sess.run(self.weights['w1'])

    def getBiases(self):  # 获取隐含层的偏执系数b1.

        return self.sess.run(self.weights['b1'])

#利用TensorFlow提供的读取示例数据的函数载入MNIST数据集。

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#定义一个对训练、测试数据进行标准化处理的函数。
# 标准化处理：均值为0，标准为1。即数据减去均值，再除以标准差。

def standard_scale(X_train, X_test):

    preprocessor = prep.StandardScaler().fit(X_train)  #sklearn.preprocessing.StandardScaler().fit()用以计算均值和标准差为后标定
    # 训练集和测试集使用相同的Scaler
    X_train = preprocessor.transform(X_train)

    X_test = preprocessor.transform(X_test)

    return X_train, X_test

#抽样数据，不放回抽样？
def get_random_block_from_data(data, batch_size):

    start_index = np.random.randint(0, len(data) - batch_size)

    return data[start_index:(start_index + batch_size)]

X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

n_samples = int(mnist.train.num_examples)

training_epochs = 20

batch_size = 128

display_step = 1

autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784,
                                            n_hidden=200,
                                            transfer_function=tf.nn.softplus,
                                            optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                                            scale=0.01)

for epoch in range(training_epochs):

    avg_cost = 0.

    total_batch = int(n_samples / batch_size)

# Loop over all batches

    for i in range(total_batch):

        batch_xs = get_random_block_from_data(X_train, batch_size)

        # Fit training using batch data

        cost = autoencoder.partial_fit(batch_xs)

        # Compute average loss

        avg_cost += cost / n_samples * batch_size

    # Display logs per epoch step

    if epoch % display_step == 0:

        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

#最后对训练完的模型进行性能测试。

print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))
