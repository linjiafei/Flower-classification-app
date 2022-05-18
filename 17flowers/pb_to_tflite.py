import tensorflow as tf

model_save_path = './checkpoint/flowers_mobilenetv3.ckpt/'

# Converting a tf.Keras model to a TensorFlow Lite model.
converter = tf.lite.TFLiteConverter.from_saved_model(model_save_path)
tflite_model = converter.convert()

# Save the model.
with open('flowers_mobilenetv3.tflite', 'wb') as f:
  f.write(tflite_model)