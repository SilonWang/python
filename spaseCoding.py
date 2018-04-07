# encoding:utf-8
# selfEncodingWithTF.py
import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data
 
'''
tensorflow实现自编码器，非监督学习
@author XueQiang Tong
'''
 
'''
xavier初始化器，把权重初始化在low和high范围内(满足N(0,2/Nin+Nout))
'''
def xavier_init(fan_in,fan_out,constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in,fan_out),minval=low ,maxval=high ,dtype=tf.float32)
 
'''数据零均值，特征方差归一化处理'''
def standard_scale(X_train,X_validation,X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_validation = preprocessor.transform(X_validation)
    X_test = preprocessor.transform(X_test)
    return X_train,X_validation,X_test
 
'''获取批量文本的策略'''
def get_random_block_from_data(data,batch_size):
    start_index = np.random.randint(0,len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]
 
'''定义的hidden层，数据结构本质是链表，其中n_node:本层节点数，n_input为输入节点数目'''
class NetLayer:
    def __init__(self,n_node,n_input):
        self.n_node = n_node
        self.n_input = n_input
        self.next_layer = None
 
    '''初始化权重'''
    def _initialize_weights(self):
        weights = dict()
        if self.next_layer == None:#如果是最后一层，由于只聚合不激活，全部初始化为0
            weights['w'] = tf.Variable(tf.zeros([self.n_input, self.n_node], dtype=tf.float32))
            weights['b'] = tf.Variable(tf.zeros([self.n_node], dtype=tf.float32))
        else:
            weights['w'] = tf.Variable(xavier_init(self.n_input, self.n_node))
            weights['b'] = tf.Variable(tf.zeros([self.n_node], dtype=tf.float32))
 
        self.weights = weights
        return self.weights
 
    '''递归计算各层的输出值，返回最后一层的输出值'''
    def cal_output(self,transfer,index,X,scale):
        if index == 0:
            self.output = transfer(tf.add(tf.matmul(X + scale * tf.random_normal([self.n_input]),self.weights['w']),self.weights['b']))
        else:
            if self.next_layer is not None:
                self.output = transfer(tf.add(tf.matmul(X,self.weights['w']),self.weights['b']))
            else:self.output = tf.add(tf.matmul(X,self.weights['w']),self.weights['b'])
        if self.next_layer is not None:
            return self.next_layer.cal_output(transfer,++index,self.output,scale)
        return self.output
 
    def get_weights(self):
        return self.weights['w']
 
    def get_bias(self):
        return self.weights['b']
 
'''定义的外层管理类'''
class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self,layers=[],transfer_function=tf.nn.softplus,optimizer=tf.train.AdamOptimizer(),scale=0.1):
        self.layers = []
        self.training_scale = scale
        self.scale = tf.placeholder(tf.float32)
        self.construct_network(layers)
        self._initialize_weights(self.layers)
 
        self.x = tf.placeholder(tf.float32,[None,layers[0]])
        self.reconstruction = self.layers[0].cal_output(transfer_function,0,self.x,scale)
 
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction,self.x),2.0))
        self.optimizer = optimizer.minimize(self.cost)
 
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)
 
    '''初始化各层并构建各层的关联'''
    def construct_network(self,layers):
        last_layer = None
        for i,layer in enumerate(layers):
            if i == 0:
                continue
            cur_layer = NetLayer(layer,layers[i-1])
            self.layers.append(cur_layer)
            if last_layer is not None:
                last_layer.next_layer = cur_layer
            last_layer = cur_layer
 
    '''外层调用初始化权重'''
    def _initialize_weights(self,layers):
        for i,layer in enumerate(layers):
            layer._initialize_weights()
 
    '''训练参数，并且返回损失函数节点的值'''
    def partial_fit(self,X):
        cost,opt = self.sess.run((self.cost,self.optimizer),
            feed_dict={self.x:X,self.scale:self.training_scale})
        return cost
 
    '''运行cost节点'''
    def calc_total_cost(self,X):
        return self.sess.run(self.cost,feed_dict={self.x:X,self.scale:self.training_scale})
 
    '''运行reconstruction节点'''
    def reconstruct(self,X):
        return self.sess.run(self.reconstruction,feed_dict={self.x:X,self.scale:self.training_scale})
 
    def fit(self,X_train,training_epochs,n_samples,batch_size):
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(n_samples / batch_size)
            for i in range(total_batch):
                batch_xs = get_random_block_from_data(X_train, batch_size)
                cost = self.partial_fit(batch_xs)
                avg_cost += cost / n_samples * batch_size
 
            if epoch % display_step == 0:
                print("Epoch:", "%04d" % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
 
if __name__ == '__main__':
    mnist = input_data.read_data_sets("E:\\Python35\\Lib\\site-packages\\tensorflow\\examples\\tutorials\\mnist\\MNIST_data",one_hot=True)
 
    X_train,X_validation,X_test = standard_scale(mnist.train.images,mnist.validation.images,mnist.test.images) #得到训练样本和测试样本
    n_samples = int(mnist.train.num_examples) #获取样本总数
    training_epochs = [20,40,60] #迭代次数
    list_layers = [[784,500,200,784],[784,200,200,784],[784,300,200,784]]
    batch_size = 128 #批次
    display_step = 1 #每隔一步显示损失函数
    mincost = (1 << 31) - 1.
    bestIter = 0
    best_layers = []
    bestModel = None
 
    '''训练出最优模型'''
    for epoch in training_epochs:
        for layers in list_layers:
            autoencoder = AdditiveGaussianNoiseAutoencoder(layers,transfer_function=tf.nn.softplus, optimizer=
                                                           tf.train.AdamOptimizer(learning_rate=0.001), scale=0.01)
            autoencoder.fit(X_train,training_epochs,n_samples,batch_size)
            cost = autoencoder.calc_total_cost(X_validation)
            if cost < mincost:
                mincost = cost
                bestModel = autoencoder
                bestIter = epoch
                best_layers = layers
 
    '''训练完毕后，用测试样本验证一下cost'''
    print("Total cost: " + str(bestModel.calc_total_cost(X_test)))