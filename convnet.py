import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, DropOut, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callback import TensorBoard
import pickle

import time

dense_layers = [0, 1, 2]
layers_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

x = pickle.load(open("x.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

x = x / 255.0


for dense_layer in dense_layers:
    for layers_size in layers_sizes:
        for conv_layer in conv_layers:
            NAME = f"{conv_layer}-conv-{layers_size}-nodes-{dense_layer}-dense-{int(time.time())}"
            tensorboard = TensorBoard(log_dir=f"logs/{NAME}")
            print(NAME)

            # NAME = f"Cats-vs-dog-cc-64x2-{int(time.time())}"

            model = Sequential()
            model.add(Conv2D(layers_size, (3, 3), input_shape=x.shape[1:]))
            model.add(Activation("relu"))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layers_size, (3, 3)))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            for l in range(dense_layer):
                model.add(Dense(dense_layer))
                model.add(Activation("relu"))

            model.add(Dense(1))
            model.add(Activation("sigmoid"))

            model.compile(loss="binary_crossentropy",
                          optimizer="adam", metrics=["accuracy"])

            model.fit(x, y, batch_size=32, epochs=10,
                      validation_split=0.1, callbacks=[tensorboard])
