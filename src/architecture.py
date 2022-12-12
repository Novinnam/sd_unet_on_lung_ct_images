import tensorflow as tf
import tensorflow_addons as tfa

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ConvBlock, self).__init__()
        self.first_conv = tf.keras.layers.Conv2D(filters, kernel_size=3, padding="same")
        self.first_group_norm = tfa.layers.GroupNormalization(groups=32)
        self.first_leaky_relu = tf.layers.LeakyReLU()
        self.second_conv = tf.layers.Conv2D(filters, kernel_size=3, padding="same")
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

class DownConv(tf.keras.layers.Layer):
    def __init__(self, filters):
        self.max_pool = tf.keras.layers.MaxPooling2D()
        self.avg_pool = tf.keras.layers.AveragePooling2D()
        self.concat = tf.keras.layers.Concatenate()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size=1)
        self.group_norm = tfa.layers.GroupNormalization(groups=32)
        self.leaky_relu = tf.keras.layers.LeakyReLU()
    
    def call(self, input):
        x = self.max_pool(input)
        y = self.avg_pool(input)
        z = self.concat([x, y])
        z = self.conv(z)
        z = self.group_norm(z)
        out = self.leaky_relu(z)
        return out

class UpConv(tf.keras.layers.Layer):
    def __init__(self, filters):
        self.up_sampling = tf.keras.layers.UpSampling2D(interpolation='bilinear')
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size=1)
        self.group_norm = tfa.layers.GroupNormalization(groups=32)
        self.leaky_relu = tf.keras.layers.LeakyReLU()
    
    def call(self, input):
        x = self.up_sampling(input)
        x = self.conv(x)
        x = self.group_norm(x)
        out = self.leaky_relu(x)
        return out
