import cv2
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#mang CNN
model = tf.keras.models.Sequential([
	tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), input_shape = (28, 28, 1), activation = tf.nn.relu),
	tf.keras.layers.MaxPooling2D(pool_size = (2,2)), 
	tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), activation = tf.nn.relu),
	tf.keras.layers.MaxPooling2D(pool_size = (2,2)),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(500, activation = tf.nn.relu),
	tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])

model.summary()