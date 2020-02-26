import tensorflow as tf
import numpy as np

model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mse')

xs = np.array([1,2,3,4,5,6,7,8,9,10], dtype=float)
ys = np.array([2,4,6,8,10,12,14,16,19,20], dtype=float)

model.fit(xs, ys, epochs=300)
print(model.predict([3.0]))

tf.keras.models.save_model(model,'m.h5')

conv = tf.lite.TFLiteConverter.from_keras_model(model)
tfl_model = conv.convert()
open('m.tflite','wb').write(tfl_model)

print('---')

interpreter = tf.lite.Interpreter(model_path="m.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_data = np.array(np.array([[3]]), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
