import tensorflow as tf
from PIL import Image
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD, PMSprop
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet121

###################################################################
# 训练参数
p_batch_size = 8
p_epochs = 50
NUM_CLASSES = 17

# 网络模型选择
model_chioce= 3
# 1:ResNet50
# 2:VGG16
# 3:MobileNetV2
# 4:InceptionV3
# 5:DenseNet121

# 是否迁移学习
transfer_lerning="imagenet"
# "imagenet"
# None

# 模型保存为h5文件
model_ResNet50="model_ResNet50.h5"
model_VGG16="model_VGG16.h5"
model_MobileNetV2="model_MobileNetV2.h5"
model_InceptionV3="model_InceptionV3.h5"
model_DenseNet121="model_DenseNet121.h5"

# 训练集和测试集位置
train_txt = './17flowers/jpg/train.txt'
x_train_savepath = './x_train.npy'
y_train_savepath = './y_train.npy'

test_txt = './17flowers/jpg/test.txt'
x_test_savepath = './x_test.npy'
y_test_savepath = './y_test.npy'

###################################################################

# ---------------------------------------------------------------------#
#   train_gpu   训练用到的GPU
#               默认为第一张卡、双卡为[0, 1]、三卡为[0, 1, 2]
#               在使用多GPU时，每个卡上的batch为总batch除以卡的数量。
# ---------------------------------------------------------------------#
train_gpu = [0, ]
# ------------------------------------------------------#
#   设置用到的显卡
# ------------------------------------------------------#
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in train_gpu)  # 修改系统环境变量；[‘环境变量名’]= ‘0’
ngpus_per_node = len(train_gpu)  # 确定使用的GPU模块个数；如 ngpus_per_node = 1
print('Number of devices: {}'.format(ngpus_per_node))  # 输出设备的个数；如 Number of devices: 1

# ------------------------------------------------------#
# np.set_printoptions(precision=None, threshold=None,  linewidth=None, suppress=None, formatter=None)
# 1.precision：控制输出结果的精度(即小数点后的位数)，默认值为8
# 2.threshold：当数组元素总数过大时，设置显示的数字位数，其余用省略号代替(当数组元素总数大于设置值，控制输出值得个数为6个，当数组元素小于或者等于设置值得时候，全部显示)，当设置值为sys.maxsize(需要导入sys库)，则会输出所有元素
# 3.linewidth：每行字符的数目，其余的数值会换到下一行
# 4.suppress：小数是否需要以科学计数法的形式输出
# 5.formatter：自定义输出规则
# ------------------------------------------------------#
np.set_printoptions(threshold=np.inf) #用于控制Python中小数的显示精度。

# 将图片制作成数据集
def generateds(txt):
    f = open(txt, 'r')  # 以只读形式打开txt文件
    contents = f.readlines()  # 读取文件中所有行
    f.close()  # 关闭txt文件
    x, y_ = [], []  # 建立空列表
    for content in contents:  # 逐行取出
        value = content.split()  # 以空格分开，图片路径为value[0] , 标签为value[1] , 存入列表
        img_path =value[0]  # 拼出图片路径和文件名
        img = Image.open(img_path)  # 读入图片
        img = img.resize((224, 224))  # 将图片尺寸调整为224*224
        img = np.array(img.convert('RGB'))  # 图片变为3*8位真彩色的np.array格式
        img = img / 255.  # 数据归一化 （实现预处理）
        x.append(img)  # 归一化后的数据，贴到列表x
        y_.append(value[1])  # 标签贴到列表y_
        print('loading : ' + content)  # 打印状态提示

    x = np.array(x)  # 变为np.array格式
    y_ = np.array(y_)  # 变为np.array格式
    y_ = y_.astype(np.int64)  # 变为64位整型
    return x, y_  # 返回输入特征x，返回标签y_

# 加载或者生成数据集
if os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath) and os.path.exists(
        x_test_savepath) and os.path.exists(y_test_savepath):
    print('-------------Load Datasets-----------------')
    x_train_save = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    x_test_save = np.load(x_test_savepath)
    y_test = np.load(y_test_savepath)
    x_train = np.reshape(x_train_save, (len(x_train_save), 224, 224, 3))
    x_test = np.reshape(x_test_save, (len(x_test_save), 224, 224, 3))
else:
    print('-------------Generate Datasets-----------------')
    x_train, y_train = generateds(train_txt)
    x_test, y_test = generateds(test_txt)

    print('-------------Save Datasets-----------------')
    x_train_save = np.reshape(x_train, (len(x_train), -1))
    x_test_save = np.reshape(x_test, (len(x_test), -1))
    np.save(x_train_savepath, x_train_save)
    np.save(y_train_savepath, y_train)
    np.save(x_test_savepath, x_test_save)
    np.save(y_test_savepath, y_test)

# 网络模型建立
if model_chioce == 1 :
    if os.path.exists(model_ResNet50):
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
    else:
        base_model = ResNet50(
            include_top=False,
            weights=transfer_lerning,
            input_shape=(224, 224, 3))
        # 全连接层
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(NUM_CLASSES, activation='softmax')(x)
        # 模型网络定义
        model = Model(inputs=base_model.input, outputs=predictions)

elif model_chioce == 2 :
    if os.path.exists(model_VGG16):
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
    else:
        base_model = VGG16(
            include_top=False,
            weights=transfer_lerning,
            input_shape=(224, 224, 3))
        # 全连接层
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(NUM_CLASSES, activation='softmax')(x)
        # 模型网络定义
        model = Model(inputs=base_model.input, outputs=predictions)

elif model_chioce == 3 :
    if os.path.exists(model_MobileNetV2):
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
    else:
        base_model = MobileNetV2(
            include_top=False,
            weights=transfer_lerning,
            input_shape=(224, 224, 3))
        # 全连接层
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(NUM_CLASSES, activation='softmax')(x)
        # 模型网络定义
        model = Model(inputs=base_model.input, outputs=predictions)

elif model_chioce == 4 :
    if os.path.exists(model_InceptionV3):
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
    else:
        base_model = InceptionV3(
            include_top=False,
            weights=transfer_lerning,
            input_shape=(224, 224, 3))
        # 全连接层
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(NUM_CLASSES, activation='softmax')(x)
        # 模型网络定义
        model = Model(inputs=base_model.input, outputs=predictions)

elif model_chioce == 5 :
    if os.path.exists(model_DenseNet121):
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
    else:
        base_model = DenseNet121(
            include_top=False,
            weights=transfer_lerning,
            input_shape=(224, 224, 3))
        # 全连接层
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(NUM_CLASSES, activation='softmax')(x)
        # 模型网络定义
        model = Model(inputs=base_model.input, outputs=predictions)


from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
if ngpus_per_node > 1:  # 如果使用的GPU模块不止一个
    model = multi_gpu_model(model, gpus=ngpus_per_node)
else:
    model = model  # 当GPU的数量是一个，model = ‘已经加载了参数的模型体’；如 ‘Mobilenet’

model.compile(
    optimizer=Adam(),# Adam(),# SGD(),# PMSprop(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["sparse_categorical_accuracy"]
)

# model.summary()
print("模型网络定义：{}层".format(len(model.layers)))

# EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    verbose=1
)

# Reduce Learning Rate
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=3,
    verbose=1
)


# 图片增强训练
# 准备图片：ImageDataGenerator
train_gen = ImageDataGenerator(
    rescale=1. / 1.,  # 如为图像，分母为255时，可归至0～1
    rotation_range=45,  # 随机45度旋转
    width_shift_range=.10,  # 宽度偏移
    height_shift_range=.10,  # 高度偏移
    horizontal_flip=True,  # 水平翻转
    zoom_range=0.1  # 将图像随机缩放阈量50％
)
test_gen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True)

# 数据集前计算
for data in (train_gen, test_gen):
    data.fit(x_train)

'''
# 实现断点续训
checkpoint_save_path = "./checkpoint/flowers_mobilenetv2.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=False,
                                                 save_best_only=True)
'''

history = model.fit(
    # x_train, y_train, batch_size=p_batch_size, # 不是数据增强
    train_gen.flow(x_train, y_train, batch_size=p_batch_size),   # 数据增强
    epochs=p_epochs,
    steps_per_epoch=x_train.shape[0] // p_batch_size,
    # validation_data=(x_test, y_test),  # 不是数据增强
    validation_data=test_gen.flow(x_test, y_test, batch_size=p_batch_size),   # 数据增强
    validation_steps=x_test.shape[0] // p_batch_size,
    callbacks=[reduce_lr]) # early_stopping,

# 打印可训练参数
#print(model.trainable_variables)
# 将可训练参数保存到文本中
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()

###############################################    show   ###############################################
# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

###################################################################
# 模型保存
if model_chioce == 1 :
    model.save("model_ResNet50.h5")
elif model_chioce == 2 :
    model.save("model_VGG16.h5")
elif model_chioce == 3:
    model.save("model_MobileNetV2.h5")
elif model_chioce == 4 :
    model.save("model_InceptionV3.h5")
elif model_chioce == 5:
    model.save("model_DenseNet121.h5")

###################################################################
# 结果评价
test_loss, test_acc = model.evaluate(
    test_gen.flow(x_test, y_test, batch_size=p_batch_size),
    steps=10)
print('val_loss: {:.3f}\nval_acc: {:.3f}'.format(test_loss, test_acc))