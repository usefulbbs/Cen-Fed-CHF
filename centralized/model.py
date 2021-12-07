import os
from keras.models import Sequential,Model
from keras.layers import Input,Dense,Flatten,Dropout,BatchNormalization,GlobalAveragePooling1D,AveragePooling1D,concatenate,add
from keras.layers import Conv1D,MaxPooling1D,Dense,UpSampling1D,ZeroPadding1D,Dropout,Reshape,multiply,Lambda
from keras.layers.core import Activation
import keras
import tensorflow as tf
import keras.backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def gelu(x):
    """基于Tanh近似计算的gelu函数
    """
    cdf = 0.5 * (1.0 + K.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * K.pow(x, 3)))))
    return x * cdf
custom_objects = {
    'gelu': gelu
}
keras.utils.get_custom_objects().update(custom_objects)

def slice_backend(x,nb_fileter,index1,index2):
    slice_number = 16
    y = x[:, :, (index1)*slice_number:(index2)*slice_number]
    return y

def bulid_MultiScaleUnet2_Model():
    def squeeze_excite_block(input_x, out_dim, ratio=4):
        squeeze = GlobalAveragePooling1D()(input_x)
        excitation = Dense(out_dim // ratio)(squeeze)
        excitation = Lambda(gelu)(excitation)
        excitation = Dense(out_dim, activation='sigmoid')(excitation)  # relu
        excitation = Reshape((1, out_dim))(excitation)
        x = multiply([input_x, excitation])
        return x
    def Conv_res(x,nb_fileter,kernerl_size,name = None,padding = None,activation = None):
        x = Conv1D(nb_fileter,kernerl_size,padding='same')(x)
        x = Lambda(gelu)(x)
        x0 = x
        x = BatchNormalization()(x)
        x = Conv1D(nb_fileter,kernerl_size,padding='same')(x)
        x = Lambda(gelu)(x)
        x = BatchNormalization()(x)
        x = squeeze_excite_block(x, nb_fileter)
        x = add([x,x0])
        x = BatchNormalization()(x)
        return x
    def decoder_out(x,number):
        x = MaxPooling1D(number)(x)
        x = Conv1D(16,3,padding='same')(x)
        x = Lambda(gelu)(x)
        x = BatchNormalization()(x)
        return x
    def Conv_res2(x,nb_fileter,kernerl_size,name = None,padding = None,activation = None):
        x = Conv1D(nb_fileter, 3,dilation_rate=1,padding='same')(x)
        x0 = x
        x1 = Lambda(slice_backend,arguments={'nb_fileter': nb_fileter,'index1':0,'index2':3})(x)
        x2 = Lambda(slice_backend, arguments={'nb_fileter': nb_fileter, 'index1': 3,'index2':6})(x)
        x3 = Lambda(slice_backend, arguments={'nb_fileter': nb_fileter, 'index1': 6,'index2':8})(x)
        y1 = x1

        y2 = Conv1D(48, 3, padding='same')(x2)
        y2 = Lambda(gelu)(y2)
        y2 = BatchNormalization()(y2)

        y3 = Conv1D(48, 3, padding='same')(x3)
        y3 = Lambda(gelu)(y3)
        y3 = BatchNormalization()(y3)



        y = concatenate([y1,y2,y3],axis=2)
        y = Conv1D(nb_fileter, 3,dilation_rate=1,padding='same')(y)
        y = Lambda(gelu)(y)
        y = BatchNormalization()(y)
        y = squeeze_excite_block(y,nb_fileter)

        out = add([x0,y])
        out = BatchNormalization()(out)
        return out
    input = Input(shape = (2000,1),name='input')
    x = input

    encoder1_1 = Conv_res(x,24,3, padding='same', activation='relu',name='encoder1_1')#(input)
    pool1 = MaxPooling1D(2,name='pool1')(encoder1_1)#30

    encoder2_1 = Conv_res(pool1,48, 3, padding='same', activation='relu',name='encoder2_1')#(pool1)
    pool2 = MaxPooling1D(2,name='pool2')(encoder2_1)#150

    up1_2 = UpSampling1D(2,name='unpool1_2')(encoder2_1)
    encoder1_2 = concatenate([encoder1_1,up1_2],axis=2,name = 'merge1')
    encoder1_2 = Conv_res(encoder1_2,24,3,padding='same',activation='relu',name='encoder1_2')#(encoder1_2)

    encoder3_1 = Conv_res(pool2,32,3,padding='same',activation='relu',name='encoder3_1')#(pool2)
    pool3 = MaxPooling1D(2,name='pool3')(encoder3_1)

    up2_2 = UpSampling1D(2,name = 'unpool2_2')(encoder3_1)
    encoder2_2 = concatenate([encoder2_1,up2_2],axis=2,name = 'merge2')
    encoder2_2 = Conv_res(encoder2_2,48,3,padding='same',activation='relu',name = 'encoder2_2')#(encoder2_2)

    up1_3 = UpSampling1D(2,name = 'unpool1_3')(encoder2_2)
    encoder1_3 = concatenate([encoder1_2,encoder1_1,up1_3],name = 'merge1_3',axis=2)
    encoder1_3 = Conv_res(encoder1_3,24,3,padding='same',activation='relu',name='encoder1_3')#(encoder1_3)

    encoder4_1 = Conv_res(pool3,16,3,padding='same',activation='relu',name = 'encoder4_1')#(pool3)
    pool4 = MaxPooling1D(2,name='pool4')(encoder4_1)

    up3_2 = UpSampling1D(2,name='unpool3_2')(encoder4_1)
    encoder3_2 = concatenate([encoder3_1,up3_2],name = 'merge3_2',axis = 2)
    encoder3_2 = Conv_res(encoder3_2,32,3,padding='same',activation='relu',name = 'encoder3_2')#(encoder3_2)

    up2_3 = UpSampling1D(2,name='unpool2_3')(encoder3_2)
    encoder2_3 = concatenate([encoder2_2,encoder2_1,up2_3],name = 'merge2_3',axis=2)
    encoder2_3 = Conv_res(encoder2_3,48,3,padding='same',activation='relu',name = 'encoder2_3')#(encoder2_3)

    up1_4 = UpSampling1D(2,name = 'unpool1_4')(encoder2_3)
    encoder1_4 = concatenate([encoder1_3,encoder1_2,encoder1_1,up1_4],name = 'merge1_4',axis = 2)
    encoder1_4 = Conv_res(encoder1_4,24,3,padding='same',activation='relu',name = 'encoder1_4')#(encoder1_4)

    # encoder5_1 = Conv_res(pool4,8,3,padding='same',activation='relu',name = 'encoder5_1')#(pool4)

    # decoder1_1 = decoder_out(encoder1_1, 8)
    decoder1_2 = decoder_out(encoder1_2, 8)
    decoder1_3 = decoder_out(encoder1_3, 8)
    decoder1_4 = decoder_out(encoder1_4, 8)

    decoder2_3  =decoder_out(encoder2_3,4)
    decoder3_2 = decoder_out(encoder3_2,2)
    decoder4_1 = Conv1D(16, 3, padding='same')(encoder4_1)
    decoder4_1 = Lambda(gelu)(decoder4_1)
    decoder4_1 = BatchNormalization()(decoder4_1)

    decoder3_1 = decoder_out(encoder3_1, 2)
    decoder2_1 = decoder_out(encoder2_1,4)



    out = concatenate([decoder1_2,decoder1_3,decoder1_4,decoder2_3,decoder3_2,decoder4_1,decoder3_1,decoder2_1],axis=2)
    out = Conv_res2(out,128,3)


    # output1 = Conv1D(1,3,activation='sigmoid',padding='same',name = 'output_1',kernel_initializer='he_normal',
    #                  kernel_regularizer=l2(1e-4))(encoder1_2)
    # output2 = Conv1D(1,3,activation='sigmoid',padding='same',name = 'output_2',kernel_initializer='he_normal',
    #                  kernel_regularizer=l2(1e-4))(encoder1_3)
    # output3 = Conv1D(1,3,activation='sigmoid',padding='same',name = 'output_3',kernel_initializer='he_normal',
    #                  kernel_regularizer=l2(1e-4))(encoder1_4)
    # output4 = Conv1D(1,3,activation='sigmoid',padding='same',name = 'output_4',kernel_initializer='he_normal')(encoder1_5)#记得取消正则
    # conv_fuse = concatenate([up2,up3,up4,up5],axis=2)
    # output5 = Conv1D(1,3,activation='sigmoid',padding='same',name='output_5',kernel_initializer='he_normal',
    #                  kernel_regularizer=l2(1e-4))(conv_fuse)
    def last_step(input):
        outputx = GlobalAveragePooling1D()(input)
        outputx = Dense(128)(outputx)
        outputx = Activation('sigmoid')(outputx)
        outputx = Dense(1)(outputx)
        outputx = Activation('sigmoid')(outputx)
        return outputx
    # output11 = last_step(output1,name='output_11')
    # output22 = last_step(output2,name='output_22')
    # output33 = last_step(output3,name='output_33')
    # output44 = last_step(encoder1_5,name='output_44')#####改了这里
    out = last_step(out)
    model1 = Model(inputs=input,outputs=out)
    model1.summary()
    return model1



