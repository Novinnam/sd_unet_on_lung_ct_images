import tensorflow as tf
import tensorflow_addons as tfa

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ConvBlock, self).__init__()
        self.first_conv = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=(1, 1), padding="same", use_bias=True)
        self.first_group_norm = tfa.layers.GroupNormalization(groups=32)
        self.first_leaky_relu = tf.layers.LeakyReLU()
        self.second_conv = tf.layers.Conv2D(filters, kernel_size=3, strides=(1, 1), padding="same", use_bias=True)
        self.second_group_norm = tfa.layers.GroupNormalization(groups=32)
        self.second_leaky_relu = tf.layers.LeakyReLU()

    def call(self, input):
        x = self.first_conv(input)
        x = self.first_group_norm(x)
        x = self.first_leaky_relu(x)
        x = self.second_conv(x)
        x = self.second_group_norm(x)
        out = self.second_leaky_relu(x)
        return out

class SaBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(SaBlock, self).__init__()
        self.x_conv_block = ConvBlock(filters)
        self.avg_pool = tf.keras.layers.AveragePooling2D()
        self.y_conv_block = ConvBlock(filters)
        self.up_sampling = tf.keras.layers.UpSampling2D()
        self.multiply = tf.keras.layers.Multiply()
        self.add = tf.keras.layers.Add()

    def call(self, input):
        x = self.x_conv_block(input)
        y = self.avg_pool(input)
        y = self.y_conv_block(y)
        y = self.up_sampling(y)
        mul = self.multiply([x, y])
        out = self.add([y, mul])
        return out



