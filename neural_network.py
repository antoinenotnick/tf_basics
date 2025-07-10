import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Flatten number data from 28x28 to 1x784 #################################################
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0

# Sequential API (Very convenient, not very flexible) #################################################
model = keras.Sequential(
    [
        keras.Input(shape=(28 * 28)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10),

    ]
)

'''
model = keras.Sequential()
model.add(keras.Input(shape=(784)))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(10))
'''

'''
model = keras.Model(inputs=model.inputs,
                    outputs=[model.layers[-1].output])
feature = model.predict(x_train)

Useful for debugging
'''

# Functional API (A bit more flexible) #################################################
inputs = keras.Input(shape=(784))
x = layers.Dense(512, activation='relu', name='first_layer')(inputs)
x = layers.Dense(256, activation='relu', name='second_layer')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

print(model.summary())

# Compiles data #################################################

model.compile(  # specifies network config (loss function, opt, etc.)
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),  # learning_rate replaces lr
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2) # v = 2 prints after each epoch, otherwise only progress bar

# print(model.summary()) is a useful debugging tool for larger models
