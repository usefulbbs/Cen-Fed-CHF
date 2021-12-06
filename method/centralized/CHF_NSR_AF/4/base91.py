import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import csv
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Model
from keras.layers import Input,Dense,Flatten,Dropout,BatchNormalization,GlobalAveragePooling1D,AveragePooling1D,concatenate,add
from keras import optimizers,regularizers,initializers
from keras.layers import Conv1D,MaxPooling1D,Dense,UpSampling1D,ZeroPadding1D,Dropout,Reshape,multiply,Lambda
from keras.layers.core import Activation
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping,ModelCheckpoint
#from squeeze_excitation import squeeze_excite_block
from keras.regularizers import l2
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
from keras.callbacks import ReduceLROnPlateau,LearningRateScheduler
import pickle
from sklearn.model_selection import StratifiedKFold
import keras
from keras.utils import plot_model
import tensorflow as tf
import keras.backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)
import numpy as np

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
def bulid_Model():
    def squeeze_excite_block(input_x, out_dim, ratio=4, name=None):
        # out_dim = tf.shape(input_x)[0]
        # se_shape = (1,out_dim)

        squeeze = GlobalAveragePooling1D()(input_x)
        # squeeze = tf.Session().run(squeeze)
        # squeeze = Reshape(se_shape)(squeeze)
        excitation = Dense(out_dim // ratio)(squeeze)
        excitation = Lambda(gelu)(excitation)
        excitation = Dense(out_dim, activation='sigmoid')(excitation)  # relu
        # excitation = Permute((2,1))(excitation)
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
        # x = Dropout(0.1)(x)
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
        #x4 = Lambda(slice_backend, arguments={'nb_fileter': nb_fileter, 'index': 4})(x)
        y1 = x1

        y2 = Conv1D(48, 3, padding='same')(x2)
        y2 = Lambda(gelu)(y2)
        y2 = BatchNormalization()(y2)

        #x3 = add([y2,x3])
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
    # x=ZeroPadding1D(8)(input)
    x=input

    encoder1_1 = Conv_res(x,24,3, padding='same', activation='relu',name='encoder1_1')#(input)
    pool1 = MaxPooling1D(2,name='pool1')(encoder1_1)#30

    encoder2_1 = Conv_res(pool1,48, 3, padding='same', activation='relu',name='encoder2_1')#(pool1)
    # encoder2_1 = Dropout(0.2,name = 'dropout1')(encoder2_1)
    pool2 = MaxPooling1D(2,name='pool2')(encoder2_1)#150

    up1_2 = UpSampling1D(2,name='unpool1_2')(encoder2_1)
    encoder1_2 = concatenate([encoder1_1,up1_2],axis=2,name = 'merge1')
    encoder1_2 = Conv_res(encoder1_2,24,3,padding='same',activation='relu',name='encoder1_2')#(encoder1_2)
    # encoder1_2 = squeeze_excite_block(encoder1_2,24)

    encoder3_1 = Conv_res(pool2,32,3,padding='same',activation='relu',name='encoder3_1')#(pool2)
    # encoder3_1 = Dropout(0.2,name = 'dropout2')(encoder3_1)
    pool3 = MaxPooling1D(2,name='pool3')(encoder3_1)

    up2_2 = UpSampling1D(2,name = 'unpool2_2')(encoder3_1)
    encoder2_2 = concatenate([encoder2_1,up2_2],axis=2,name = 'merge2')
    encoder2_2 = Conv_res(encoder2_2,48,3,padding='same',activation='relu',name = 'encoder2_2')#(encoder2_2)

    up1_3 = UpSampling1D(2,name = 'unpool1_3')(encoder2_2)
    encoder1_3 = concatenate([encoder1_2,encoder1_1,up1_3],name = 'merge1_3',axis=2)
    encoder1_3 = Conv_res(encoder1_3,24,3,padding='same',activation='relu',name='encoder1_3')#(encoder1_3)
    # encoder1_3 = squeeze_excite_block(encoder1_3,24)

    encoder4_1 = Conv_res(pool3,16,3,padding='same',activation='relu',name = 'encoder4_1')#(pool3)
    # encoder4_1 = Dropout(0.2,name = 'dropout3')(encoder4_1)
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
    # encoder1_4 = squeeze_excite_block(encoder1_4,24)

    encoder5_1 = Conv_res(pool4,8,3,padding='same',activation='relu',name = 'encoder5_1')#(pool4)
    # encoder5_1 = Dropout(0.2,name = 'dropout4')(encoder5_1)

    decoder1_1 = decoder_out(encoder1_1, 8)
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

    '''
    up4_2 = UpSampling1D(2,name = 'unpool4_2')(encoder5_1)
    encoder4_2 = concatenate([up4_2,encoder4_1],name = 'merge4_2',axis = 2)
    encoder4_2 = Conv_res(encoder4_2,16,3,padding='same',activation='relu',name = 'encoder4_2')#(encoder4_2)

    up3_3 = UpSampling1D(2,name='unpool3_3')(encoder4_2)
    encoder3_3 = concatenate([up3_3,encoder3_1,encoder3_2],name = 'merge3_3',axis = 2)
    encoder3_3 = Conv_res(encoder3_3,32,3,padding='same',activation='relu',name = 'encoder3_3')#(encoder3_3)

    up2_4 = UpSampling1D(2,name = 'unpool2_4')(encoder3_3)
    encoder2_4 = concatenate([up2_4,encoder2_1,encoder2_2,encoder2_3],name = 'merge2_4',axis = 2)
    encoder2_4 = Conv_res(encoder2_4,48,3,padding='same',activation='relu',name = 'encoder2_4')#(encoder2_4)

    up1_5 = UpSampling1D(2,name = 'unpool1_5')(encoder2_4)
    encoder1_5 = concatenate([up1_5,encoder1_1,encoder1_2,encoder1_3,encoder1_4],name = 'merge1_5',axis = 2)
    encoder1_5 = Conv_res(encoder1_5,24,3,padding='same',activation='relu',name = 'encoder1_5')#(encoder1_5)
    # encoder1_5 = squeeze_excite_block(encoder1_5,24)
    '''

    output1 = Conv1D(1,3,activation='sigmoid',padding='same',name = 'output_1',kernel_initializer='he_normal',
                     kernel_regularizer=l2(1e-4))(encoder1_2)
    output2 = Conv1D(1,3,activation='sigmoid',padding='same',name = 'output_2',kernel_initializer='he_normal',
                     kernel_regularizer=l2(1e-4))(encoder1_3)
    output3 = Conv1D(1,3,activation='sigmoid',padding='same',name = 'output_3',kernel_initializer='he_normal',
                     kernel_regularizer=l2(1e-4))(encoder1_4)
    #output4 = Conv1D(1,3,activation='sigmoid',padding='same',name = 'output_4',kernel_initializer='he_normal')(encoder1_5)#记得取消正则
    # conv_fuse = concatenate([up2,up3,up4,up5],axis=2)
    # output5 = Conv1D(1,3,activation='sigmoid',padding='same',name='output_5',kernel_initializer='he_normal',
    #                  kernel_regularizer=l2(1e-4))(conv_fuse)
    def last_step(input,name = None):
        outputx = GlobalAveragePooling1D()(input)
        outputx = Dense(128)(outputx)
        outputx = Activation('sigmoid')(outputx)
        outputx = Dense(1)(outputx)
        outputx = Activation('sigmoid')(outputx)
        return outputx
    output11 = last_step(output1,name='output_11')
    output22 = last_step(output2,name='output_22')
    output33 = last_step(output3,name='output_33')
    #output44 = last_step(encoder1_5,name='output_44')#####改了这里
    out = last_step(out)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=5,mode = 'auto',factor=0.1)
    # model1 = Model(inputs=input,outputs=[output11,output22,output33,output44])
    model1 = Model(inputs=input,outputs=out)
    model1.summary()
    return model1  # ,reduce_lr

def remake_data(label,counts,total_data,total_labels):
    temp_data = np.zeros(shape=(1,2000))
    temp_lables = np.zeros(shape=(1,1))
    for i in range(len(total_data)):
        cur_data = total_data[i]
        cur_lable = total_labels[i]
        cur_data = np.reshape(cur_data,newshape=(1,2000))
        cur_lable = np.reshape(cur_lable,newshape=(1,1))
        if total_labels[i] == label:
            temp_data = np.concatenate([temp_data,cur_data],0)
            temp_lables = np.concatenate([temp_lables,cur_lable],0)
            counts -= 1
        if counts == 0:
            break
    return temp_data[1:,:],temp_lables[1:,:]
def count_label_number(labels):
    temp_ones_count = 0
    temp_zeros_count = 0
    for i in labels:
        if i == 1:
            temp_ones_count +=1
        if i == 0:
            temp_zeros_count += 1
    return [temp_zeros_count,temp_ones_count]

for i in [4]:
    K.clear_session()

    train_chf_data = pd.read_csv('/data/wanghd/wanghong_s1/2000_10/'+str(i)+'/train/chf/all.csv')
    test_chf_data = pd.read_csv('/data/wanghd/wanghong_s1/2000_10/'+str(i)+'/test/chf/all.csv')
    train_nsr_data = pd.read_csv('/data/wanghd/wanghong_s1/2000_10/'+str(i)+'/train/nsr/all.csv')
    test_nsr_data = pd.read_csv('/data/wanghd/wanghong_s1/2000_10/'+str(i)+'/test/nsr/all.csv')
    val_chf_data = pd.read_csv('/data/wanghd/wanghong_s1/2000_10/' + str(i) + '/val/chf/all.csv')
    val_nsr_data = pd.read_csv('/data/wanghd/wanghong_s1/2000_10/' + str(i) + '/val/nsr/all.csv')

    train_af_data = pd.read_csv('/data/wanghd/wanghong_s1/AF_data/'+str(i)+'/'+'AF_train_'+str(i)+'1.csv')
    train_val_af_data = pd.read_csv('/data/wanghd/wanghong_s1/AF_data/'+str(i)+'/'+'AF_val_'+str(i)+'1.csv')
    train_af_data = np.array(train_af_data)
    train_val_af_data = np.array(train_val_af_data)
    af_data = np.concatenate([train_af_data,train_val_af_data],0)
    af_labels = np.zeros(shape=(af_data.shape[0],1))

    test_af_data = pd.read_csv('/data/wanghd/wanghong_s1/AF_data/'+str(i)+'/test/all.csv')
    test_af_data = np.array(test_af_data)
    test_af_labels = np.zeros(shape=(test_af_data.shape[0],1))
    '''
    train_chf_data = pd.read_csv('F:/硕-联邦学习/wanghd_s1/2000_10/' + str(i) + '/train/chf/all.csv')
    test_chf_data = pd.read_csv('F:/硕-联邦学习/wanghd_s1/2000_10/' + str(i) + '/test/chf/all.csv')
    train_nsr_data = pd.read_csv('F:/硕-联邦学习/wanghd_s1/2000_10/' + str(i) + '/train/nsr/all.csv')
    test_nsr_data = pd.read_csv('F:/硕-联邦学习/wanghd_s1/2000_10/' + str(i) + '/test/nsr/all.csv')
    val_chf_data = pd.read_csv('F:/硕-联邦学习/wanghd_s1/2000_10/' + str(i) + '/val/chf/all.csv')
    val_nsr_data = pd.read_csv('F:/硕-联邦学习/wanghd_s1/2000_10/' + str(i) + '/val/nsr/all.csv')
    '''
    train_chf_label = np.ones(shape=(train_chf_data.shape[0],1))
    test_chf_label = np.ones(shape=(test_chf_data.shape[0],1))
    train_nsr_label = np.zeros(shape=(train_nsr_data.shape[0],1))
    val_chf_label = np.ones(shape=(val_chf_data.shape[0],1))
    val_nsr_label = np.zeros(shape=(val_nsr_data.shape[0],1))
    test_nsr_label = np.zeros(shape=(test_nsr_data.shape[0],1))
    train_data=np.concatenate([train_chf_data,train_nsr_data], 0)
    train_label=np.concatenate([train_chf_label,train_nsr_label], 0)
    test_data = np.concatenate([test_chf_data, test_nsr_data], 0)
    test_label = np.concatenate([test_chf_label, test_nsr_label], 0)
    train_data = np.concatenate([train_data, test_data], 0)
    train_label = np.concatenate([train_label, test_label], 0)
    val_data = np.concatenate([val_chf_data, val_nsr_data], 0)
    val_label = np.concatenate([val_chf_label, val_nsr_label], 0)

    train_count = count_label_number(train_label)
    print('train_count:', train_count)
    val_count = count_label_number(val_label)
    print('val_count:', val_count)

    train_data = np.concatenate([train_data, af_data], 0)
    train_label = np.concatenate([train_label, af_labels], 0)
    val_data = np.concatenate([val_data, test_af_data], 0)
    val_label = np.concatenate([val_label, test_af_labels], 0)


    np.random.seed(2)
    np.random.shuffle(train_data)
    np.random.seed(2)
    np.random.shuffle(train_label)
    np.random.seed(2)
    np.random.shuffle(val_data)
    np.random.seed(2)
    np.random.shuffle(val_label)

    train_ones_data,train_ones_lables = remake_data(1,train_count[1],train_data,train_label)
    train_zeros_data,train_zeros_lables = remake_data(0,train_count[0],train_data,train_label)
    val_ones_data,val_ones_labels = remake_data(1,val_count[1],val_data,val_label)
    val_zeros_data,val_zeros_lables = remake_data(0,val_count[0],val_data,val_label)

    train_data = np.concatenate([train_ones_data,train_zeros_data], 0)
    train_label = np.concatenate([train_ones_lables, train_zeros_lables], 0)
    val_data = np.concatenate([val_ones_data, val_zeros_data], 0)
    val_label = np.concatenate([val_ones_labels,val_zeros_lables], 0)

    np.random.seed(2)
    np.random.shuffle(train_data)
    np.random.seed(2)
    np.random.shuffle(train_label)
    np.random.seed(2)
    np.random.shuffle(val_data)
    np.random.seed(2)
    np.random.shuffle(val_label)

    # train_data,val_data,train_label,val_label=train_test_split(train_val_data, train_val_label, test_size=0.15)
    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
    val_data = val_data.reshape(val_data.shape[0], val_data.shape[1], 1)
    # train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
    # val_data = val_data.reshape(val_data.shape[0], val_data.shape[1], 1)
    # model1, reduce_lr = bulid_Model()
    model1 = bulid_Model()
    model1.compile(optimizer=optimizers.Adam(lr=1e-4), loss='categorical_hinge', metrics=['accuracy'])
    mc = ModelCheckpoint(filepath='cnn_ed_val2000' + str(i) + '.h5', monitor='val_acc', mode='max', verbose=2,save_best_only=False)
    # model1 = model1.fit(train_data,[train_label,train_label,train_label,train_label],epochs = 30,batch_size=8,validation_data=(val_data,
    # [val_label,val_label,val_label,val_label]),verbose=1,callbacks=[mc])
    def scheduler(epoch):
        if epoch < 10:
            LR = 0.0001
            print('lr:',LR)
            K.set_value(model1.optimizer.lr, LR)
            return K.get_value(model1.optimizer.lr)
        elif 10 <= epoch <20:
            LR = 0.00001
            print('lr:', LR)
            K.set_value(model1.optimizer.lr, LR)
            return K.get_value(model1.optimizer.lr)
        elif 20 <= epoch < 25:
            LR = 0.000001
            print('lr:',LR)
            K.set_value(model1.optimizer.lr, LR)
            return K.get_value(model1.optimizer.lr)
        elif epoch >= 25:
            LR = K.get_value(model1.optimizer.lr)
            LR = LR*0.1
            K.set_value(model1.optimizer.lr, LR)
            return K.get_value(model1.optimizer.lr)
    reducue_lr = LearningRateScheduler(scheduler)
    model1 = model1.fit(train_data, train_label, epochs=60, batch_size=16, validation_data=(val_data, val_label),
                        verbose=2, callbacks=[mc,reducue_lr])

    with open('trainHistoryDict' + str(i) + '.txt', 'wb') as file_pi:
        pickle.dump(model1.history, file_pi)

    fig = plt.figure()
    fig.tight_layout()  # 调整整体空白
    plt.subplots_adjust(wspace=0.4, hspace=0.5)  # 调整子图间距
    # pl.xlim(0, 1)  # 限定横轴的范围

    # plt.ylim(0.6, 1)  # 限定纵轴的范围
    # plt.xticks([0,10,20,30,40,50,60,70,80,90,100])
    plt.legend(['train'], loc='best')
    plt.subplot(221)
    # plt.plot(model1.history['activation_2_acc'],color = 'blue',label = 'activation_2_acc',linewidth = 0.5)
    # plt.plot(model1.history['activation_4_acc'],color = 'g',label = 'activation_4_acc',linewidth = 0.5)
    # plt.plot(model1.history['activation_6_acc'],color = 'gold',label = 'activation_6_acc',linewidth = 0.5)
    plt.plot(model1.history['acc'], color='gold', label='Train Accuracy', linewidth=2)

    plt.title('(a)Train Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid(linestyle='-.')
    plt.legend()
    plt.subplot(222)
    # plt.plot(model1.history['val_acc'],color = 'g',linewidth = '2')
    # plt.plot(model1.history['val_activation_2_acc'],color = 'blue',label = 'val_activation_2_acc',linewidth = 0.5)
    # plt.plot(model1.history['val_activation_4_acc'],color = 'g',label = 'val_activation_4_acc',linewidth = 0.5)
    # plt.plot(model1.history['val_activation_6_acc'],color = 'gold',label = 'val_activation_6_acc',linewidth = 0.5)
    plt.plot(model1.history['val_acc'], color='gold', label='Validation Accuracy', linewidth=2)

    plt.title('(b)Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid(linestyle='-.')

    plt.legend()
    plt.subplot(223)
    # plt.plot(model1.history['loss'],color = 'gold',linewidth = '2') #plt.plot(history.history['val_loss'])
    # plt.plot(model1.history['activation_2_loss'],color = 'blue',label = 'activation_2_loss',linewidth = 0.5)
    # plt.plot(model1.history['activation_4_loss'],color = 'g',label = 'activation_4_loss',linewidth = 0.5)
    # plt.plot(model1.history['activation_6_loss'],color = 'gold',label = 'activation_6_loss',linewidth = 0.5)
    plt.plot(model1.history['loss'], color='gold', label='Train Loss', linewidth=2)

    plt.title('(c)Train Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(linestyle='-.')

    plt.legend()
    plt.subplot(224)
    # plt.plot(model1.history['val_loss'],color = 'y',linewidth = '2')
    # plt.plot(model1.history['val_activation_2_loss'],color = 'blue',label = 'val_activation_2_loss',linewidth = 0.5)
    # plt.plot(model1.history['val_activation_4_loss'],color = 'g',label = 'val_activation_4_loss',linewidth = 0.5)
    # plt.plot(model1.history['val_activation_6_loss'],color = 'gold',label = 'val_activation_6_loss',linewidth = 0.5)
    plt.plot(model1.history['val_loss'], color='gold', label='Validation Loss', linewidth=2)

    plt.title('(d)Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(linestyle='-.')
    # plt.xticks([0,10,20,30,40,50,60,70,80,90,100])
    # plt.ylim(0, 1)

    plt.legend()
    plt.savefig(str(i) + 'ACCURACY2000.png', format='png')
    # plt.show()

    validation_accuracy = model1.history['val_acc']
    df1 = pd.DataFrame(validation_accuracy)
    df1.to_csv('validation_acc' + str(i), index=None)
path_dir ='/data/wanghd/wanghong_s1/change1_c_hinge/'
k_mean = 0
'''
for k in range(1,11):
    filename = path_dir + 'validation_acc' + str(k)
    record = pd.read_csv(filename)
    record = np.array(record, dtype=np.float32)
    n = len(record)
    one_mean = (np.mean(record[n-1:n]))
    print(record[n-1:n])
    print(one_mean)
    k_mean = one_mean + k_mean
print(k_mean/10)
'''