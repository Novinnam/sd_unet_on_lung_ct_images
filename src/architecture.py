import tensorflow as tf

class SaBlock(tf.keras.layers.Layer):
    def __init__(self, c):
        super(SaBlock, self).__init__()

        self.c = c
        self.conv_1 = tf.keras.layers.Conv2D(self.c, )

    def call(self, x):
