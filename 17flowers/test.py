import tensorflow as tf
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet121
import class_labels

np.set_printoptions(threshold=np.inf)  # 用于控制Python中小数的显示精度。
'''
np.set_printoptions(precision=None, threshold=None,  linewidth=None, suppress=None, formatter=None)
1.precision：控制输出结果的精度(即小数点后的位数)，默认值为8
2.threshold：当数组元素总数过大时，设置显示的数字位数，其余用省略号代替(当数组元素总数大于设置值，控制输出值得个数为6个，当数组元素小于或者等于设置值得时候，全部显示)，当设置值为sys.maxsize(需要导入sys库)，则会输出所有元素
3.linewidth：每行字符的数目，其余的数值会换到下一行
4.suppress：小数是否需要以科学计数法的形式输出
5.formatter：自定义输出规则
'''

###################################################################
# 训练参数
p_batch_size = 8
p_epochs = 10
NUM_CLASSES = 17
###################################################################

model_chioce= 3
# 1:ResNet50
# 2:VGG16
# 3:MobileNetV2
# 4:InceptionV3
# 5:DenseNet121

model_ResNet50="model_ResNet50.h5"
model_VGG16="model_VGG16.h5"
model_MobileNetV2="model_MobileNetV2.h5"
model_InceptionV3="model_InceptionV3.h5"
model_DenseNet121="model_DenseNet121.h5"


if model_chioce == 1 :
    base_model = ResNet50(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3))
    # 全连接层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(model_ResNet50)  # 给模型加载模型参数

elif model_chioce == 2 :
    base_model = VGG16(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3))
    # 全连接层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(model_VGG16)  # 给模型加载模型参数

elif model_chioce == 3 :
    base_model = MobileNetV2(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3))
    # 全连接层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(model_MobileNetV2)  # 给模型加载模型参数

elif model_chioce == 4 :
    base_model = InceptionV3(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3))
    # 全连接层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(model_InceptionV3)  # 给模型加载模型参数

elif model_chioce == 5 :
    base_model = DenseNet121(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3))
    # 全连接层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(model_DenseNet121)  # 给模型加载模型参数



# 预测图片数量
preNum = 29
picture_path=["buttercup.jpg","buttercup1.jpg","buttercup2.jpg","buttercup2.jpg","colts_foot.jpg","colts_foot2.jpg","colts_foot3.jpg",
      "cowslip.jpg","cowslip1.jpg","cowslip2.jpg","crocus.jpg","crocus1.jpg","crocus2.jpg","daffodil.jpg","daffodil1.jpg","daffodil2.jpg",
      "dandelion.jpg","dandelion1.jpg","dandelion2.jpg","dandelion3.jpg","pansy.jpg","snowdrop.jpg","snowdrop1.jpg","tigerlily1.jpg","tigerlily2.jpg",
      "windflower.jpg","windflower1.jpg","windflower2.jpg","windflower3.jpg"]
# preNum = int(input("input the number of test pictures:"))
# 预测图片路径
for i in range(preNum):
    image_path ="./test/"+picture_path[i]
    # image_path ="./test/"+input("the path of test picture:")
    # 打开图片
    img = Image.open(image_path)

    # 显示图片
    # image = plt.imread(image_path)
    #plt.set_cmap('gray')
    # plt.imshow(image)

    # 调整尺寸和类型
    img = img.resize((224, 224), Image.ANTIALIAS)
    img_arr = np.array(img.convert('RGB'))
    img_arr = img_arr / 255.0   # 数据归一化 （实现预处理）
    x_predict = img_arr[tf.newaxis, ...]
    # 预测
    result = model.predict(x_predict)
    #print(result)

    #显示预测值前五的结果
    top5_index = np.argsort(result[0])[::-1][0:5]
    print("检测图片："+picture_path[i])
    print("检测结果如下：")
    for i in top5_index:
        tf.print("  "+class_labels.labels[i] + ":" + str(result[0][i]))
    print('\n')

    # plt.pause(1)
    # plt.close()