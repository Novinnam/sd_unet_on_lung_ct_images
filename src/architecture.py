import tensorflow as tf
import tensorflow_addons as tfa

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, dilation_rate=1):
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.layers.Conv2D(filters, kernel_size=kernel_size,
                                           padding='same', 
                                           dilation_rate=dilation_rate)
        self.group_norm = tfa.layers.GroupNormalization(groups=32)

    def call(self, input):
        x = self.conv(input)
        x = self.group_norm(x)
        out = tf.nn.leaky_relu(x)
        return out

class SaBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(SaBlock, self).__init__()
        self.first_x_conv_block = ConvBlock(filters)
        self.second_x_conv_block = ConvBlock(filters)
        self.first_y_conv_block = ConvBlock(filters)
        self.second_y_conv_block = ConvBlock(filters)
        self.up_sampling = tf.keras.layers.UpSampling2D()

    def call(self, input):
        x = self.first_x_conv_block(input)
        x = self.second_x_conv_block(x)
        y = tf.nn.avg_pool2d(input)
        y = self.first_y_conv_block(y)
        y = self.second_y_conv_block(y)
        y = self.up_sampling(y)
        mul = tf.multiply(x, y)
        out = tf.add(y, mul)
        return out

class DownConvBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        self.conv_block = ConvBlock(filters, kernel_size=1)
    
    def call(self, input):
        x = tf.nn.max_pool2d(input)
        y = tf.nn.avg_pool2d(input)
        z = tf.concat([x, y], axis=-1)
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

class DenseAsppBlock(tf.keras.layers.Layer):
    def __init__(self, filters):
        self.second_branch_conv_block = ConvBlock(filters, dilation_rate=18)
        self.third_branch_conv_block = ConvBlock(filters, dilation_rate=12)
        self.fourth_branch_conv_block = ConvBlock(filters, dilation_rate=6)
        self.fifth_branch_conv_block = ConvBlock(filters, kernel_size=1)
        self.last_branch_conv_block = ConvBlock(filters, kernel_size=1)
    
    def call(self, input):
        dims = input.shape
        fifth_branch = self.fourth_conv_block(input)
        fourth_branch = self.fourth_branch_conv_block(input)
        third_branch_concat = tf.concat([input, fourth_branch], axis=-1)
        third_branch = self.third_branch_conv_block(third_branch_concat)
        second_branch_first_concat = tf.concat([input, third_branch_concat], axis=-1)
        second_branch_second_concat = tf.concat([third_branch, second_branch_first_concat], axis=-1)
        second_branch = self.second_branch_conv_block(second_branch_second_concat)
        first_branch = tf.nn.avg_pool2d(input=input, ksize=(dims[-3, dims[-2]]))

        out = tf.concat([first_branch, second_branch, third_branch, fourth_branch, fifth_branch], axis=-1)
        return out
        
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