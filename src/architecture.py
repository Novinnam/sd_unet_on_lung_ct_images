import tensorflow as tf
import tensorflow_addons as tfa

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, dilation_rate=1):
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size,
                                           padding='same', 
                                           dilation_rate=dilation_rate)
        self.group_norm = tfa.layers.GroupNormalization(groups=32)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.group_norm(x)
        out = tf.nn.leaky_relu(x)
        return out

class SaBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(SaBlock, self).__init__()
        self.first_x_conv_block = ConvBlock(filters)
        self.second_x_conv_block = ConvBlock(filters)
        self.avg_pool = tf.keras.layers.AveragePooling2D()
        self.first_y_conv_block = ConvBlock(filters)
        self.second_y_conv_block = ConvBlock(filters)
        self.up_sampling = tf.keras.layers.UpSampling2D()

    def call(self, inputs):
        x = self.first_x_conv_block(inputs)
        x = self.second_x_conv_block(x)
        y = self.avg_pool(inputs)
        y = self.first_y_conv_block(y)
        y = self.second_y_conv_block(y)
        y = self.up_sampling(y)
        mul = tf.multiply(x, y)
        out = tf.add(y, mul)
        return out

class DownConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(DownConvBlock, self).__init__()
        self.avg_pool = tf.keras.layers.AveragePooling2D()
        self.max_pool = tf.keras.layers.MaxPooling2D()
        self.conv_block = ConvBlock(filters, kernel_size=1)
    
    def call(self, inputs):
        x = self.avg_pool(inputs)
        y = self.max_pool(inputs)
        z = tf.concat([x, y], axis=-1)
        out = self.conv_block(z)
        return out

class UpConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(UpConvBlock, self).__init__()
        self.up_sampling = tf.keras.layers.UpSampling2D(interpolation='bilinear')
        self.conv_block = ConvBlock(filters, kernel_size=1)
    
    def call(self, inputs):
        x = self.up_sampling(inputs)
        out = self.conv_block(x)
        return out    

class DenseAsppBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(DenseAsppBlock, self).__init__()
        self.second_branch_conv_block = ConvBlock(filters, dilation_rate=18)
        self.third_branch_conv_block = ConvBlock(filters, dilation_rate=12)
        self.fourth_branch_conv_block = ConvBlock(filters, dilation_rate=6)
        self.fifth_branch_conv_block = ConvBlock(filters, kernel_size=1)
        self.last_branch_conv_block = ConvBlock(filters, kernel_size=1)
        self.avg_pool = tf.keras.layers.AveragePooling2D()
    
    def call(self, inputs):
        fifth_branch = self.fifth_branch_conv_block(inputs)
        fourth_branch = self.fourth_branch_conv_block(inputs)
        third_branch_concat = tf.concat([inputs, fourth_branch], axis=-1)
        third_branch = self.third_branch_conv_block(third_branch_concat)
        second_branch_first_concat = tf.concat([inputs, third_branch_concat], axis=-1)
        second_branch_second_concat = tf.concat([third_branch, second_branch_first_concat], axis=-1)
        second_branch = self.second_branch_conv_block(second_branch_second_concat)
        first_branch = self.avg_pool(inputs)

        concat = tf.concat([first_branch, second_branch, third_branch, fourth_branch, fifth_branch], axis=-1)
        out = self.last_branch_conv_block(concat)
        return out
        
class SdUnet(tf.keras.models.Model):
    def __init__(self):
        super(SdUnet, self).__init__()
        # Encoder
        self.first_encode_sa_block = SaBlock(32)
        self.first_down_conv_block = DownConvBlock(32)
        self.second_encode_sa_block = SaBlock(64)
        self.second_down_conv_block = DownConvBlock(64)
        self.third_encode_sa_block = SaBlock(128)
        self.third_down_conv_block = DownConvBlock(128)
        self.fourth_encode_sa_block = SaBlock(256)
        self.fourth_down_conv_block = DownConvBlock(256)
        self.fifth_encode_sa_block = SaBlock(512)
        # BottleNeck
        self.dense_aspp_block = DenseAsppBlock(256)
        # Decoder
        self.first_up_conv_block = UpConvBlock(256)
        self.first_decode_sa_block = SaBlock(256)
        self.second_up_conv_block = UpConvBlock(128)
        self.second_decode_sa_block = SaBlock(128)
        self.third_up_conv_block = UpConvBlock(64)
        self.third_decode_sa_block = SaBlock(64)
        self.fourth_up_conv_block = UpConvBlock(32)
        self.fourth_decode_sa_block = SaBlock(32)
    
    def call(self, inputs):
        # Encoder
        first_sa = self.first_encode_sa_block(inputs)
        x = self.first_down_conv_block(first_sa)
        second_sa = self.second_encode_sa_block(x)
        x = self.second_down_conv_block(second_sa)
        third_sa = self.third_encode_sa_block(x)
        x = self.third_down_conv_block(third_sa)
        fourth_sa = self.fourth_encode_sa_block(x)
        x = self.fourth_down_conv_block(fourth_sa)
        fifth_sa = self.fifth_encode_sa_block(x)
        # BottleNeck
        x = self.dense_aspp_block(fifth_sa)
        # Decoder
        first_up = self.first_up_conv_block(x)
        x_concat = tf.concat([fourth_sa, first_up], axis=-1)
        x = self.first_decode_sa_block(x_concat)
        second_up = self.second_up_conv_block(x)
        x_concat = tf.concat([third_sa, second_up], axis=-1)
        x = self.second_decode_sa_block(x_concat)
        third_up = self.third_up_conv_block(x)
        x_concat = tf.concat([second_sa, third_up], axis=-1)
        x = self.third_decode_sa_block(x_concat)
        fourth_up = self.fourth_up_conv_block(x)
        x_concat = tf.concat([first_sa, fourth_up], axis=-1)
        out = self.fourth_decode_sa_block(x_concat)
        return out