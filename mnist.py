from keras.datasets import mnist
import matplotlib.pyplot as plt
#加载数据
(X_train, Y_train ),(X_test , Y_test) = mnist.load_data()
#展示第一张图
plt.imshow(X_train[16], cmap= plt.get_cmap('PuBuGn_r'))
plt.show()