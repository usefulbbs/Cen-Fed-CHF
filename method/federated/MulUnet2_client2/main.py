import nest_asyncio
nest_asyncio.apply()

import os
import collections
import time
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.simulation import hdf5_client_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
from sklearn.metrics import roc_curve, auc
from matplotlib.pyplot import MultipleLocator


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling1D, AveragePooling1D, concatenate, add, Reshape
from tensorflow.keras.layers import Conv1D, MaxPooling1D,  UpSampling1D, ZeroPadding1D,multiply,Lambda
from tensorflow.keras import optimizers, regularizers, initializers
from tensorflow.keras.layers import Activation
from tensorflow.keras.regularizers import l2

from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras import backend as K


NUM_EPOCHS = 1
BATCH_SIZE = 16
SHUFFLE_BUFFER = 50
PREFETCH_BUFFER = 4096
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def map_fn(example):
  return collections.OrderedDict(
      x=tf.reshape(example['pixels'], [-1, 2000]), y=example['label'])


def client_data(n):
  ds = emnist_train.create_tf_dataset_for_client(n)
  return ds.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).map(map_fn).prefetch(PREFETCH_BUFFER)


def create_keras_model():
    def gelu(x):
        """基于Tanh近似计算的gelu函数
        """
        cdf = 0.5 * (1.0 + K.tanh(
            (np.sqrt(2 / np.pi) * (x + 0.044715 * K.pow(x, 3)))))
        return x * cdf

    def slice_backend(x, nb_fileter, index1, index2):
        slice_number = 16
        y = x[:, :, (index1) * slice_number:(index2) * slice_number]
        return y

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

    def Conv_res(x, nb_fileter, kernerl_size, name=None, padding=None, activation=None):
        x = Conv1D(nb_fileter, kernerl_size, padding='same')(x)
        x = Lambda(gelu)(x)
        x0 = x
        x = BatchNormalization()(x,training=True)
        x = Conv1D(nb_fileter, kernerl_size, padding='same')(x)
        x = Lambda(gelu)(x)
        # x = Dropout(0.1)(x)
        x = BatchNormalization()(x,training=True)
        x = squeeze_excite_block(x, nb_fileter)
        x = add([x, x0])
        x = BatchNormalization()(x,training=True)
        return x

    def decoder_out(x, number):
        x = MaxPooling1D(number)(x)
        x = Conv1D(16, 3, padding='same')(x)
        x = Lambda(gelu)(x)
        x = BatchNormalization()(x,training=True)
        return x

    def Conv_res2(x, nb_fileter, kernerl_size, name=None, padding=None, activation=None):
        x = Conv1D(nb_fileter, 3, dilation_rate=1, padding='same')(x)
        x0 = x
        x1 = Lambda(slice_backend, arguments={'nb_fileter': nb_fileter, 'index1': 0, 'index2': 3})(x)
        x2 = Lambda(slice_backend, arguments={'nb_fileter': nb_fileter, 'index1': 3, 'index2': 6})(x)
        x3 = Lambda(slice_backend, arguments={'nb_fileter': nb_fileter, 'index1': 6, 'index2': 8})(x)
        # x4 = Lambda(slice_backend, arguments={'nb_fileter': nb_fileter, 'index': 4})(x)
        y1 = x1

        y2 = Conv1D(48, 3, padding='same')(x2)
        y2 = Lambda(gelu)(y2)
        y2 = BatchNormalization()(y2,training=True)

        # x3 = add([y2,x3])
        y3 = Conv1D(48, 3, padding='same')(x3)
        y3 = Lambda(gelu)(y3)
        y3 = BatchNormalization()(y3,training=True)

        y = concatenate([y1, y2, y3], axis=2)
        y = Conv1D(nb_fileter, 3, dilation_rate=1, padding='same')(y)
        y = Lambda(gelu)(y)
        y = BatchNormalization()(y,training=True)
        y = squeeze_excite_block(y, nb_fileter)

        out = add([x0, y])
        out = BatchNormalization()(out,training=True)
        return out

    input = Input(shape=(2000, 1), name='input')
    # x=ZeroPadding1D(8)(input)
    x = input

    encoder1_1 = Conv_res(x, 24, 3, padding='same', activation='relu', name='encoder1_1')  # (input)
    pool1 = MaxPooling1D(2, name='pool1')(encoder1_1)  # 30

    encoder2_1 = Conv_res(pool1, 48, 3, padding='same', activation='relu', name='encoder2_1')  # (pool1)
    # encoder2_1 = Dropout(0.2,name = 'dropout1')(encoder2_1)
    pool2 = MaxPooling1D(2, name='pool2')(encoder2_1)  # 150

    up1_2 = UpSampling1D(2, name='unpool1_2')(encoder2_1)
    encoder1_2 = concatenate([encoder1_1, up1_2], axis=2, name='merge1')
    encoder1_2 = Conv_res(encoder1_2, 24, 3, padding='same', activation='relu', name='encoder1_2')  # (encoder1_2)
    # encoder1_2 = squeeze_excite_block(encoder1_2,24)

    encoder3_1 = Conv_res(pool2, 32, 3, padding='same', activation='relu', name='encoder3_1')  # (pool2)
    # encoder3_1 = Dropout(0.2,name = 'dropout2')(encoder3_1)
    pool3 = MaxPooling1D(2, name='pool3')(encoder3_1)

    up2_2 = UpSampling1D(2, name='unpool2_2')(encoder3_1)
    encoder2_2 = concatenate([encoder2_1, up2_2], axis=2, name='merge2')
    encoder2_2 = Conv_res(encoder2_2, 48, 3, padding='same', activation='relu', name='encoder2_2')  # (encoder2_2)

    up1_3 = UpSampling1D(2, name='unpool1_3')(encoder2_2)
    encoder1_3 = concatenate([encoder1_2, encoder1_1, up1_3], name='merge1_3', axis=2)
    encoder1_3 = Conv_res(encoder1_3, 24, 3, padding='same', activation='relu', name='encoder1_3')  # (encoder1_3)
    # encoder1_3 = squeeze_excite_block(encoder1_3,24)

    encoder4_1 = Conv_res(pool3, 16, 3, padding='same', activation='relu', name='encoder4_1')  # (pool3)
    # encoder4_1 = Dropout(0.2,name = 'dropout3')(encoder4_1)
    pool4 = MaxPooling1D(2, name='pool4')(encoder4_1)

    up3_2 = UpSampling1D(2, name='unpool3_2')(encoder4_1)
    encoder3_2 = concatenate([encoder3_1, up3_2], name='merge3_2', axis=2)
    encoder3_2 = Conv_res(encoder3_2, 32, 3, padding='same', activation='relu', name='encoder3_2')  # (encoder3_2)

    up2_3 = UpSampling1D(2, name='unpool2_3')(encoder3_2)
    encoder2_3 = concatenate([encoder2_2, encoder2_1, up2_3], name='merge2_3', axis=2)
    encoder2_3 = Conv_res(encoder2_3, 48, 3, padding='same', activation='relu', name='encoder2_3')  # (encoder2_3)

    up1_4 = UpSampling1D(2, name='unpool1_4')(encoder2_3)
    encoder1_4 = concatenate([encoder1_3, encoder1_2, encoder1_1, up1_4], name='merge1_4', axis=2)
    encoder1_4 = Conv_res(encoder1_4, 24, 3, padding='same', activation='relu', name='encoder1_4')  # (encoder1_4)
    # encoder1_4 = squeeze_excite_block(encoder1_4,24)

    encoder5_1 = Conv_res(pool4, 8, 3, padding='same', activation='relu', name='encoder5_1')  # (pool4)
    # encoder5_1 = Dropout(0.2,name = 'dropout4')(encoder5_1)

    decoder1_1 = decoder_out(encoder1_1, 8)
    decoder1_2 = decoder_out(encoder1_2, 8)
    decoder1_3 = decoder_out(encoder1_3, 8)
    decoder1_4 = decoder_out(encoder1_4, 8)

    decoder2_3 = decoder_out(encoder2_3, 4)
    decoder3_2 = decoder_out(encoder3_2, 2)
    decoder4_1 = Conv1D(16, 3, padding='same')(encoder4_1)
    decoder4_1 = Lambda(gelu)(decoder4_1)
    decoder4_1 = BatchNormalization()(decoder4_1,training=True)

    decoder3_1 = decoder_out(encoder3_1, 2)
    decoder2_1 = decoder_out(encoder2_1, 4)

    out = concatenate([decoder1_2, decoder1_3, decoder1_4, decoder2_3, decoder3_2, decoder4_1, decoder3_1, decoder2_1],
                      axis=2)
    out = Conv_res2(out, 128, 3)

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

    output1 = Conv1D(1, 3, activation='sigmoid', padding='same', name='output_1', kernel_initializer='he_normal',
                     kernel_regularizer=l2(1e-4))(encoder1_2)
    output2 = Conv1D(1, 3, activation='sigmoid', padding='same', name='output_2', kernel_initializer='he_normal',
                     kernel_regularizer=l2(1e-4))(encoder1_3)
    output3 = Conv1D(1, 3, activation='sigmoid', padding='same', name='output_3', kernel_initializer='he_normal',
                     kernel_regularizer=l2(1e-4))(encoder1_4)

    # output4 = Conv1D(1,3,activation='sigmoid',padding='same',name = 'output_4',kernel_initializer='he_normal')(encoder1_5)#记得取消正则
    # conv_fuse = concatenate([up2,up3,up4,up5],axis=2)
    # output5 = Conv1D(1,3,activation='sigmoid',padding='same',name='output_5',kernel_initializer='he_normal',
    #                  kernel_regularizer=l2(1e-4))(conv_fuse)
    def last_step(input, name=None):
        outputx = GlobalAveragePooling1D()(input)
        outputx = Dense(128)(outputx)
        outputx = Activation('sigmoid')(outputx)
        outputx = Dense(1)(outputx)
        outputx = Activation('sigmoid')(outputx)
        return outputx

    output11 = last_step(output1, name='output_11')
    output22 = last_step(output2, name='output_22')
    output33 = last_step(output3, name='output_33')
    # output44 = last_step(encoder1_5,name='output_44')#####改了这里
    out = last_step(out)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=5,mode = 'auto',factor=0.1)
    # model1 = Model(inputs=input,outputs=[output11,output22,output33,output44])
    model = Model(inputs=input, outputs=out)
    #model1.summary()
    return model
'''
#Create your class extending the wrapper
class FocalLoss(LossFunctionWrapper):
#Implement the constructor - here you can give extended arguments to it.
   def __init__(self,
                 gamma,
                 alpha,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='binary_focal_loss'):
        super(FocalLoss, self).__init__(
            binary_focal_loss,
            name=name,
            reduction=reduction,
            gamma=gamma,
            alpha=alpha)

def binary_focal_loss(y_true, y_pred, gamma,alpha):
        ones = K.ones_like(y_true)
        alpha_t = y_true * alpha + (ones - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (ones - y_true) * (ones - y_pred) + K.epsilon()
        focal_loss = -alpha_t * K.pow((ones - p_t), gamma) * K.log(p_t)
        return K.mean(focal_loss)
binary_focal_loss = FocalLoss(gamma=2,alpha=0.25)
'''
def model_fn():
  model = create_keras_model()
  return tff.learning.from_keras_model(
      model,
      input_spec=element_spec,
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.BinaryAccuracy()])


def evaluation(state, test_data, test_label):
    keras_model = create_keras_model()
    keras_model.compile(
       loss=tf.keras.losses.MeanSquaredError(),
       metrics=[tf.keras.metrics.BinaryAccuracy()],
      )
    tff.learning.ModelWeights.assign_weights_to(state.model,keras_model)
    tf.keras.backend.set_learning_phase(1)
    print('test:')
    loss, accurcy = keras_model.evaluate(x=test_data, y=test_label, batch_size=16)
    y_pred = keras_model.predict(x=test_data)
    return loss, accurcy, y_pred


def tff_build():
    trainer = tff.learning.build_federated_averaging_process(model_fn, client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.0007)
                                                             , server_optimizer_fn=lambda: tf.keras.optimizers.Adam(decay=0.05))
    state = trainer.initialize()
    return trainer, state


def get_file(path):
    files = os.listdir(path)
    files.sort()
    list=[]
    for file in files:
        if not os.path.isdir(path+file):
            f_name = str(file)
            tr = '/'
            filename = path+tr+f_name
            list.append(filename)
    return list

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def plot_roc(labels, predict_prob):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, predict_prob)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=true_positive_rate, FPR=false_positive_rate, threshold=thresholds)
    print(thresholds)
    plt.title('ROC')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.ylabel('TPR')
    plt.xlabel('FPR')
    plt.pause(60)
    plt.savefig(name3)
    plt.close()
    return false_positive_rate, true_positive_rate, roc_auc, optimal_th, optimal_point

fold_loss = np.zeros(shape=10)
fold_accuracy = np.zeros(shape=10)
k_fold = 10
NUM_CLIENTS = 2
num_rounds = 230
for i in [NUM_CLIENTS]:
    client_num = i
    path = r'/home/b227/PycharmProjects/H_Zexin/datasetFinal/dataset2'
    print(path)
    for j in [2,3,4,5,6,7,8,9,10]:
        print('client_num:', i, 'fold:', j)
        train_filename = path + '/train_client'+str(j)+'.h5'
        data_validation_filename = path + '/data_val'+str(j)
        label_validation_filename = path + '/label_val'+str(j)
        print('train_filename:', train_filename)
        print('data_validation_filename:', data_validation_filename)
        print('label_validation_filename:', label_validation_filename)

        emnist_train = hdf5_client_data.HDF5ClientData(train_filename)  # HDF5ClientData
        sample_clients = emnist_train.client_ids[0:client_num]
        print('sample_clients:', sample_clients)
        train_data = [client_data(n) for n in sample_clients]
        print('train_data:', train_data)
        element_spec = train_data[0].element_spec
        trainer, state = tff_build()


        validation_data = pd.read_csv(data_validation_filename)
        validation_data = np.array(validation_data, dtype=np.float32)
        validation_data = validation_data[1:, :]


        validation_label = pd.read_csv(label_validation_filename)
        validation_label = np.array(validation_label, dtype=np.float32)
        validation_label = validation_label[1:, :]


        validation_data = validation_data.reshape(validation_data.shape[0], validation_data.shape[1])
        validation_label = validation_label.reshape(validation_label.shape[0], validation_label.shape[1])

        train_accuracy_record = np.zeros(shape=230)
        validation_accuracy_record = np.zeros(shape=230)
        last_5rounds_accuracy = np.zeros(shape=5)
        last_5rounds_loss = np.zeros(shape=5)
        for r in range(num_rounds):
            t1 = time.time()
            state, metrics = trainer.next(state, train_data)
            train_accuracy_record[r] = (metrics['train']['binary_accuracy'])
            # print(metrics['train']['binary_accuracy'])
            t2 = time.time()
            print('round:', r+1)
            print('metrics {m}, round time {t:.2f} seconds'.format(m=metrics, t=t2 - t1))
            validation_loss, validation_accuracy, validation_pred = evaluation(state=state, test_data=validation_data, test_label=validation_label)
            validation_accuracy_record[r] = (validation_accuracy)
            if r >= num_rounds-5:
                last_5rounds_accuracy[r-225] = (validation_accuracy)
                last_5rounds_loss[r-225] = (validation_loss)
            print('last_5rounds_accuracy:', last_5rounds_accuracy)
            print('*************************************************************************************************')
        name3 = 'roc_' + str(j) + 'fold_' + 'divide equally-client_num = 2'
        false_positive_rate, true_positive_rate, roc_auc, optimal_th, optimal_point = plot_roc(labels=validation_label, predict_prob=validation_pred)
        print('optimal_threshold:', optimal_th)
        print('optimal_point:', optimal_point)

        df1 = pd.DataFrame(validation_accuracy_record)
        df1.to_csv('validation_accuracy_record' + str(j), index=None)
        optimal = [optimal_th, optimal_point]
        df2 = pd.DataFrame(optimal)
        df2.to_csv('optimal' + str(j), index=None)

        last_5rounds_loss_mean = np.mean(last_5rounds_loss)
        last_5rounds_accuracy_mean = np.mean(last_5rounds_accuracy)
        # test_loss, test_accuracy = MAIN(trainer=trainer, state=state)
        fold_loss[j-1] = last_5rounds_loss_mean
        fold_accuracy[j-1] = last_5rounds_accuracy_mean
        print('fold_loss:', fold_loss)
        print('fold_acuuracy:', fold_accuracy)


        save_model_name = 'client_num'+str(i)+'_10fold_'+str(j)
        save_model = create_keras_model()
        tff.learning.ModelWeights.assign_weights_to(state.model,save_model)
        save_model.save(save_model_name)


        name1 = 'train_' + str(j) + 'fold_'+'divide equally-client_num = 2'
        name2 = 'val_' + str(j) + 'fold_'+'divide equally-client_num = 2'

        fig1, ax =plt.subplots()
        x = np.linspace(1, 230, 230)
        ax.plot(x, train_accuracy_record)
        plt.xlabel('rounds')
        plt.ylabel('accuracy')
        plt.savefig(name1)
        plt.pause(3)
        plt.cla()
        plt.close(fig1)

        fig2, ay = plt.subplots()
        ay.plot(x, validation_accuracy_record)
        plt.xlabel('rounds')
        plt.ylabel('accuracy')
        plt.savefig(name2)
        plt.pause(3)
        plt.cla()
        plt.close(fig2)