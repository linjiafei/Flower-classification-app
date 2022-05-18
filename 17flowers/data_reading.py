"""
pip install tflearn
数据集为17 Category Flower Dataset，是牛津大学Visual Geometry Group选取的在英国比较常见的17种花；其中每种花有80张图片，整个数据集有1360张图片；类别已经分好，标签就是最外层的文件夹的名字1-16，在输入标签的时候可以直接通过文件读取的方式。
"""

from tflearn.datasets import oxflower17
import matplotlib.pyplot as plt
import numpy as np

#如果没有数据会自动下载数据
X,Y = oxflower17.load_data(
    dirname='./17flowers', resize_pics=(224, 224), shuffle=True, one_hot=True)


print(type(X))
print("图片的数值大小:{} - {}".format(X.max(), X.min()))
print(X.shape, Y.shape)
print(type(Y))

for i in range(3):
    idx = np.random.randint(1, 1361)
    print('图片的索引号为：{} - 花的种类为:{}'.format(idx, np.argmax(Y[idx])))
    plt.imshow(X[idx])
    plt.show()
