from functools import wraps
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from utils.utils import compose


#--------------------------------------------------#
#   单次卷积
#--------------------------------------------------#
@wraps(Conv2D)                      #darknet53的卷积
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}      #正则化
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):        #卷积块
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),     #卷积
        BatchNormalization(),                       #标准化
        LeakyReLU(alpha=0.1))                       #激活函数

#---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def resblock_body(x, num_filters, num_blocks):
    x = ZeroPadding2D(((1,0),(1,0)))(x)             #零填充
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)    #卷积块，步长2
    #残差网络结构
    for i in range(num_blocks):         #重复次数
        y = DarknetConv2D_BN_Leaky(num_filters//2, (1,1))(x)    #1*1卷积，通道数1/2
        y = DarknetConv2D_BN_Leaky(num_filters, (3,3))(y)       #3*3卷积，通道数扩展回来
        x = Add()([x,y])
    return x

#---------------------------------------------------#
#   darknet53 的主体部分
#---------------------------------------------------#
def darknet_body(x):                        #darkmet53的结构
    #416*416*3
    x = DarknetConv2D_BN_Leaky(32, (3,3))(x)    #通道数调整为32
    #416*416*32
    x = resblock_body(x, 64, 1)         #输入特征层，输出通道数，重复次数
    # 208*208*64
    x = resblock_body(x, 128, 2)
    # 104*104*128
    x = resblock_body(x, 256, 8)
    feat1 = x
    #52*52*256，特征层提取
    x = resblock_body(x, 512, 8)
    feat2 = x
    #26*26*512，特征层提取
    x = resblock_body(x, 1024, 4)
    feat3 = x
    #13*13*1024，特征提取
    return feat1,feat2,feat3

