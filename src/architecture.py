import tensorflow as tf
import tensorflow_addons as tfa

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, dilation_rate=1):
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size,
                                           padding='same', 
                                           dilation_rate=dilation_rate)
        self.group_norm = tfa.layers.GroupNormalization(groups=32)
        self.leaky_relu = tf.layers.LeakyReLU()

    def call(self, input):
        x = self.conv(input)
        x = self.group_norm(x)
        out = self.leaky_relu(x)
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
        self.multiply = tf.keras.layers.Multiply()
        self.add = tf.keras.layers.Add()

    def call(self, input):
        x = self.first_x_conv_block(input)
        x = self.second_x_conv_block(x)
        y = self.avg_pool(input)
        y = self.first_y_conv_block(y)
        y = self.second_y_conv_block(y)
        y = self.up_sampling(y)
        mul = self.multiply([x, y])
        out = self.add([y, mul])
        return out

class DownConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        self.max_pool = tf.keras.layers.MaxPooling2D()
        self.avg_pool = tf.keras.layers.AveragePooling2D()
        self.concat = tf.keras.layers.Concatenate()
        self.conv_block = ConvBlock(filters, kernel_size=1)
    
    def call(self, input):
        x = self.max_pool(input)
        y = self.avg_pool(input)
        z = self.concat([x, y])
        out = self.conv_block(z)
        return out

class UpConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        self.up_sampling = tf.keras.layers.UpSampling2D(interpolation='bilinear')
        self.conv_block = ConvBlock(filters, kernel_size=1)
    
    def call(self, input):
        x = self.up_sampling(input)
        out = self.conv_block(x)
        return out

def DASPP(dspp_input):
  dims = dspp_input.shape
  x = tf.keras.layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
  print("Stage1:", x.shape)
  x = convolution_block(x, num_filters=256, kernel_size=1, dilation_rate=1, padding="same", use_bias=True)
  out_pool = tf.keras.layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]),
         interpolation="bilinear",
    )(x)

  out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
  out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
  x_12 = tf.keras.layers.Concatenate(axis=-1)([dspp_input, out_6])
  out_12 = convolution_block(x_12, kernel_size=3, dilation_rate=12)

  x_18_1 = tf.keras.layers.Concatenate(axis=-1)([dspp_input, x_12])
  x_18_2 = tf.keras.layers.Concatenate(axis=-1)([x_18_1, out_12])
  out_18 = convolution_block(x_18_2, kernel_size=3, dilation_rate=18)

  x = tf.keras.layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
  output = convolution_block(x, kernel_size=1)
  return output

def SD_UNet(input_size=(288, 288, 3)):
  input = tf.keras.Input(input_size)
  x1 = SaBlock(32, input)
  x2 = DownConvBlock(32, x1)
  print("Stage1", x2.shape)

  x3 = SaBlock(64, x2)
  x4 = DownConvBlock(64, x3)
  print("Stage2", x4.shape)

  x5 = SaBlock(128, x4)
  x6 = DownConvBlock(128, x5)
  print("Stage3", x6.shape)

  x7 = SaBlock(256, x6)
  x8 = DownConvBlock(256, x7)
  print("Stage4", x8.shape)

  x9 = SaBlock(512, x8)
  print("Stage5", x9.shape)
  
  x10 = DASPP(x9)
  print("Stage6", x10.shape)

  # Check stage7 and stage8 dimensions
  x11 = UpConvBlock(256, x10)
  print("Stage7", x11.shape)
  x12 = tf.keras.layers.Concatenate(axis=-1)([x11, x7])
  print("Stage7_1", x12.shape)
  x13 = SaBlock(256, x12)
  print("Stage8", x13.shape)

  x14 = UpConvBlock(128, x13)
  print("Stage9", x14.shape)
  x15 = tf.keras.layers.Concatenate(axis=-1)([x14, x5])
  print("Stage9_1", x15.shape)
  x16 = SaBlock(128, x15)
  print("Stage10", x16.shape)

  x17 = UpConvBlock(64, x16)
  print("Stage11", x17.shape)
  x18 = tf.keras.layers.Concatenate(axis=-1)([x17, x3])
  print("Stage11_1", x18.shape)
  x19 = SaBlock(64, x18)
  print("Stage12", x19.shape)

  x20 = UpConvBlock(32, x19)
  print("Stage13", x20.shape)
  x21 = tf.keras.layers.Concatenate(axis=-1)([x20, x1])
  print("Stage13_1", x21.shape)
  x22 = SaBlock(32, x21)
  print("Stage14", x22.shape)

  out = tf.keras.layers.Conv2D(filters=1, kernel_size=3, padding="same", use_bias=True, activation='sigmoid')(x22)
  print("Stage15", out.shape)
  return tf.keras.Model(inputs=input, outputs=out, name="SD_UNet")