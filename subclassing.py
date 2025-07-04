os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

physcical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physcical_devices[0], True)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28 * 28).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype('float32') / 255.0

# CNN -> BatchNorm -> ReLU (Common structure)
# x10

class CNNBlock(layers.Layer):
    def __init__(self, outchannels, kernel_size=3):
        super(CNNBlock, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size, padding='same')
        self.bn = layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.relu(x)
        return x
    
model = keras.Sequential(
    [
        CNNBlock(32),
        CNNBlock(64),
        CNNBlock(128),
        layers.Flatten(),
        layers.Dense(10),


    ]
)

model.compile(
    optimizer.keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=64, epochs=3, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)