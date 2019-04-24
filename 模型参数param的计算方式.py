#!/usr/bin/python3
# coding: utf-8

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import Input, Model
from keras.layers import Embedding, Bidirectional, Dense, Dropout, GRU, LSTM
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.utils import multi_gpu_model
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from keras_preprocessing.text import Tokenizer

def keras_mnist_mlp_h256():
    # 多层感知器模型
    model = Sequential()
    model.add(Dense(units=256,
                    input_dim=784,
                    kernel_initializer='normal',
                    activation='relu'))

    # input_dim=784：设置“输入层”神经元个数为784
    # （因为原本为28*28的二维图像，以reshape转换为一维的向量，也就是28*28=784个float数）
    # units=256：定义“隐藏层”神经元个数为256
    # 建立输入层与隐藏层的公式如下：
    # h1 = relu(X*W1 + b1)

    model.add(Dense(units=10,
                    kernel_initializer='normal',
                    activation='softmax'))
    # units=10： 定义“输出层”神经元个数为10
    # 此处，建立Dense神经网络层不需要设置input_dim，
    # 因为keras会自动按照上一层的units是256个神经元，设置这一层的input_dim为256个神经元；
    # 建立隐藏层与输出层的公式如下：
    # y = softmax(h1 * W2 + b2)

    print(model.summary())
    '''
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 256)               200960    = 784*256+256
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                2570      = 256*10+10
    =================================================================
    Total params: 203,530    = (784*256+256)+(256*10+10)
    Trainable params: 203,530  # 必须全部训练的参数params
    Non-trainable params: 0
    _________________________________________________________________
    None
    '''
    # 每一层param计算方式如下：
    # param = (上一层神经元数量) * (本层的神经元数量) + (本层的神经元数量)
    # 隐藏层的param是200960， 这是因为：
    # 784（输入层神经元数量）× 256（隐藏层神经元数量）+ 256（隐藏层神经元数量）= 200960
    # 输出层的param是2570，这是因为：
    # 256（隐藏层神经元数量）× 10（输出层神经元数量）+10（输出层神经元数量）=2570
    # 全部必须训练的超参数（Trainable params）是每一层的param的总和，计算公式如下：
    # 200960（隐藏层的param）+ 2570（输出层的param）=203530
    # 通常Trainable params数值越大，代表此模型越复杂，需要更多时间进行训练。

def keras_cifar_cnn():
    model = Sequential()

    # 卷积层1
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     input_shape=(32, 32, 3),
                     activation='relu',
                     padding='same'))
    # filters=32： 设置随机产生32个滤镜
    # kernel_size=(3, 3)： 每一个滤镜大小为3*3
    # padding='same'： 填充，让卷积产生的卷积图像大小不变
    # input_shape=(32, 32, 3)： 第1、2维，代表输入的图像形状大小为32*32；
    # 第3维，因为是彩色，所以代表RGB三原色是3；

    model.add(Dropout(rate=0.25))
    # 池化层1
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 没有与MaxPool layer相关的参数量.尺寸,步长和填充数都是超参数.
    # pool_size=(2, 2) 缩减采样；将32*32的图像缩小为16*16的图像；缩减采样不会改变神经元数量

    # 卷积层2
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     activation='relu', padding='same'))
    model.add(Dropout(0.25))
    # 池化层2
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # 建立神经网络(平坦层、隐藏层、输出层)
    model.add(Flatten())
    model.add(Dropout(rate=0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(10, activation='softmax'))
    print(model.summary())
    '''
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 32, 32, 32)        896       = (3*3)*3*32+32
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 32, 32, 32)        0         
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 16, 16, 32)        0         16 = 32/2
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 16, 16, 64)        18496     = (3*3)*32*64+64
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 16, 16, 64)        0         
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 8, 8, 64)          0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 4096)              0         4096 = 8*8*64
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 4096)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 1024)              4195328   = 4096*1024 + 1024
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 1024)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                10250     = 1024*10 + 10
    =================================================================
    Total params: 4,224,970                                          = 896+18496+4195328+10250
    Trainable params: 4,224,970
    Non-trainable params: 0
    _________________________________________________________________
    None
    在CNN中,每层有两种类型的参数:weights 和biases.总参数数量为所有weights和biases的总和.
    
    定义如下:
    WC=卷积层的weights数量
    BC=卷积层的biases数量
    PC=所有参数的数量
    K=核尺寸
    N=核数量
    C=输入图像通道数(或上一层神经元数量)
    卷积层中,核的深度等于输入图像的通道数.于是每个核有K*K个参数.并且有N个核.由此得出以下的公式.
    WC = K*K × C × N
    BC = N
    PC = WC + BC
    
    卷积层输入W1*H1*D1大小的数据，输出W2*H2*D2的数据，此时的卷积层共有4个超参数：
    K：滤波器个数
    P：pad属性值(填充)
    S：滤波器每次移动的步长
    F：滤波器尺寸
    此时输出的大小可以用输入和超参计算得到：
    W2=（W1-F+2P）/S+1
    H2=（H1-F+2P）/S+1
    D2=D1
    
    
    '''
def keras_cifar_cnn2():
    model = Sequential()
    # 卷积层1
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
                     input_shape=(32, 32, 3),
                     activation='relu'))
    # 因为没有填充操作，让卷积产生的卷积图像大小由 32*32的图像，在输出时变成了30*30的图像
    model.add(Dropout(rate=0.25))
    # 池化层1
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 没有与MaxPool layer相关的参数量.尺寸,步长和填充数都是超参数.
    # 缩减采样，图像大小由30*30 变成了 15*15

    # 卷积层2
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
                     activation='relu'))
    # 因没有填充操作，图像大小由输入时的15*15，在输出时变成了13*13

    model.add(Dropout(0.25))
    # 池化层2
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 建立神经网络(平坦层、隐藏层、输出层)
    model.add(Flatten())
    model.add(Dropout(rate=0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(rate=0.25))
    model.add(Dense(10, activation='softmax'))
    print(model.summary())

    '''
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 30, 30, 32)        896       = (3*3)*3*32+32
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 30, 30, 32)        0         
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 15, 15, 32)        0         # 池化操作，缩减采样图像大小由30*30变成了15*15
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 13, 13, 64)        18496     # 因没有填充操作，图像大小由15*15变成了13*13；18496 = (3*3)*32*64+64
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 13, 13, 64)        0         
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 6, 6, 64)          0         # 池化操作，缩减采样，图像大小有13*13变成了6*6
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 2304)              0         = 6*6*64
    _________________________________________________________________
    dropout_3 (Dropout)          (None, 2304)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 1024)              2360320   = 2304*1024+1024
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 1024)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                10250     = 1024*10 + 10
    =================================================================
    Total params: 2,389,962                                          = 896 + 18496 + 2360320+ 10250
    Trainable params: 2,389,962
    Non-trainable params: 0
    _________________________________________________________________
    None
    '''

def bi_lstm_crf():
    word_input = Input(shape=(None,), dtype='int32', name="word_input")
    word_emb = Embedding(6864 + 1, 300,
                         weights=None,
                         trainable=False, # 设置为False，则本层的参数是不可训练的（Non-trainable params）；
                         name='word_emb')(word_input)
    # 嵌入层Embedding 将正整数（下标）转换为具有固定大小的向量，如[[4],[20]]->[[0.25,0.1],[0.6,-0.2]]
    # 参数数量 = input_dim * output_dim

    bilstm_output = Bidirectional(LSTM(256 // 2,
                                       return_sequences=True))(word_emb)
    bilstm_output = Dropout(0.1)(bilstm_output)
    output = Dense(259 + 1, kernel_initializer="he_normal")(bilstm_output)
    output = CRF(259 + 1, sparse_target=False)(output)
    model = Model([word_input], [output])  # Model 类将输入张量和输出张量转换为一个模型
    print(model.summary())
'''

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
word_input (InputLayer)      (None, None)              0         
_________________________________________________________________
word_emb (Embedding)         (None, None, 300)         2059500   = 6865*300
_________________________________________________________________
bidirectional_1 (Bidirection (None, None, 256)         439296    
_________________________________________________________________
dropout_1 (Dropout)          (None, None, 256)         0         
_________________________________________________________________
dense_1 (Dense)              (None, None, 260)         66820     
_________________________________________________________________
crf_1 (CRF)                  (None, None, 260)         135980    
=================================================================
Total params: 2,701,596                                          = 2059500+439296+66820+135980
Trainable params: 642,096                                        = 439296+66820+135980
Non-trainable params: 2,059,500
_________________________________________________________________
None

'''
def main():
    # keras_mnist_mlp_h256()
    # keras_cifar_cnn()
    # keras_cifar_cnn2()
    bi_lstm_crf()


if __name__ == '__main__':
    main()

# TensorFlow+Keras深度学习人工智能实践应用.pdf p71-p73