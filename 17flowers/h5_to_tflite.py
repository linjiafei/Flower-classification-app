'''
import tensorflow as tf

model_savepath="model.h5"

converter = tf.lite.TFLiteConverter.from_keras_model(model_savepath)
tflite_model = converter.convert()
open("model.tflite", "wb").write(tflite_model)
'''
import tensorflow as tf
# 将h5模型转化为tflite模型方法1
modelparh = r"model_MobileNetV2.h5"
model = tf.keras.models.load_model(modelparh)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
savepath = r"model.tflite"
open(savepath, "wb").write(tflite_model)
