import os
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping,ModelCheckpoint
import tensorflow as tf
import keras.backend as K
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau,LearningRateScheduler
from .model import bulid_MultiScaleUnet2_Model
from .utility import count_label_number,remake_data
import pickle
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
config = tf.ConfigProto()
config.gpu_options.allow_growth = False
config.gpu_options.per_process_gpu_memory_fraction = 0.2
sess = tf.Session(config=config)

epochs_lr4 = 10
epochs_lr5 = 10
epochs_lr6 = 5
#  The decay period of the learning rate can be obtained according to internal cross-validation
cur_Kfold = [7]
for i in cur_Kfold:
    K.clear_session()
    train_chf_data = pd.read_csv('/data/external/'+str(i)+'/train/chf/all.csv')
    test_chf_data = pd.read_csv('/data/external/'+str(i)+'/test/chf/all.csv')
    train_nsr_data = pd.read_csv('/data/external/'+str(i)+'/train/nsr/all.csv')
    test_nsr_data = pd.read_csv('/data/external/'+str(i)+'/test/nsr/all.csv')
    val_chf_data = pd.read_csv('/data/external/' + str(i) + '/val/chf/all.csv')
    val_nsr_data = pd.read_csv('/data/external/' + str(i) + '/val/nsr/all.csv')

    # train_af_data = pd.read_csv('/data/wanghd/wanghong_s1/AF_data/'+str(i)+'/'+'AF_train_'+str(i)+'1.csv')
    # train_val_af_data = pd.read_csv('/data/wanghd/wanghong_s1/AF_data/'+str(i)+'/'+'AF_val_'+str(i)+'1.csv')
    # train_af_data = np.array(train_af_data)
    # train_val_af_data = np.array(train_val_af_data)
    # af_data = np.concatenate([train_af_data,train_val_af_data],0)
    # af_labels = np.zeros(shape=(af_data.shape[0],1))
    #
    # train_ap_data = pd.read_csv('/data/wanghd/wanghong_s1/Apnea_data/'+str(i)+'/'+'train_train_'+str(i)+'1.csv')
    # train_val_ap_data = pd.read_csv('/data/wanghd/wanghong_s1/Apnea_data/'+str(i)+'/'+'train_val_'+str(i)+'1.csv')
    # train_ap_data = np.array(train_ap_data)
    # train_val_ap_data = np.array(train_val_ap_data)
    # ap_data = np.concatenate([train_ap_data,train_val_ap_data],0)
    # ap_labels = np.zeros(shape=(ap_data.shape[0],1))


    # test_af_data = pd.read_csv('/data/wanghd/wanghong_s1/AF_data/'+str(i)+'/test/all.csv')
    # test_af_data = np.array(test_af_data)
    # test_af_labels = np.zeros(shape=(test_af_data.shape[0],1))
    #
    # test_ap_data = pd.read_csv('/data/wanghd/wanghong_s1/Apnea_data/'+str(i)+'/test/test_all.csv')
    # test_ap_data = np.array(test_ap_data)
    # test_ap_labels = np.zeros(shape=(test_ap_data.shape[0],1))

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

    # train_count = count_label_number(train_label)
    # print('train_count:', train_count)
    # val_count = count_label_number(val_label)
    # print('val_count:', val_count)
    #
    # train_data = np.concatenate([train_data, af_data,ap_data], 0)
    # train_label = np.concatenate([train_label, af_labels,ap_labels], 0)
    # val_data = np.concatenate([val_data, test_af_data,test_ap_data], 0)
    # val_label = np.concatenate([val_label, test_af_labels,test_ap_labels], 0)


    np.random.seed(2)
    np.random.shuffle(train_data)
    np.random.seed(2)
    np.random.shuffle(train_label)
    np.random.seed(2)
    np.random.shuffle(val_data)
    np.random.seed(2)
    np.random.shuffle(val_label)

    # train_ones_data,train_ones_lables = remake_data(1,train_count[1],train_data,train_label)
    # train_zeros_data,train_zeros_lables = remake_data(0,train_count[0],train_data,train_label)
    # val_ones_data,val_ones_labels = remake_data(1,val_count[1],val_data,val_label)
    # val_zeros_data,val_zeros_lables = remake_data(0,val_count[0],val_data,val_label)

    # train_data = np.concatenate([train_ones_data,train_zeros_data], 0)
    # train_label = np.concatenate([train_ones_lables, train_zeros_lables], 0)
    # val_data = np.concatenate([val_ones_data, val_zeros_data], 0)
    # val_label = np.concatenate([val_ones_labels,val_zeros_lables], 0)

    # np.random.seed(2)
    # np.random.shuffle(train_data)
    # np.random.seed(2)
    # np.random.shuffle(train_label)
    # np.random.seed(2)
    # np.random.shuffle(val_data)
    # np.random.seed(2)
    # np.random.shuffle(val_label)

    train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
    val_data = val_data.reshape(val_data.shape[0], val_data.shape[1], 1)

    model1 = bulid_MultiScaleUnet2_Model()
    model1.compile(optimizer=optimizers.Adam(lr=1e-4), loss='categorical_hinge', metrics=['accuracy'])
    mc = ModelCheckpoint(filepath='cnn_ed_val2000' + str(i) + '.h5', monitor='val_acc', mode='max', verbose=2,save_best_only=False)

    def scheduler(epoch):
        if epoch < epochs_lr4:
            LR = 0.0001
            print('lr:',LR)
            K.set_value(model1.optimizer.lr, LR)
            return K.get_value(model1.optimizer.lr)
        elif epochs_lr4 <= epoch <(epochs_lr4+epochs_lr5):
            LR = 0.00001
            print('lr:', LR)
            K.set_value(model1.optimizer.lr, LR)
            return K.get_value(model1.optimizer.lr)
        elif (epochs_lr4+epochs_lr5) <= epoch < (epochs_lr4+epochs_lr5+epochs_lr6):
            LR = 0.000001
            print('lr:',LR)
            K.set_value(model1.optimizer.lr, LR)
            return K.get_value(model1.optimizer.lr)
        elif epoch >= (epochs_lr4+epochs_lr5+epochs_lr6):
            LR = K.get_value(model1.optimizer.lr)
            LR = LR*0.1
            K.set_value(model1.optimizer.lr, LR)
            return K.get_value(model1.optimizer.lr)
    reducue_lr = LearningRateScheduler(scheduler)
    model1 = model1.fit(train_data, train_label, epochs=60, batch_size=16, validation_data=(val_data, val_label),
                        verbose=2, callbacks=[mc,reducue_lr])

    with open('trainHistoryDict' + str(i) + '.txt', 'wb') as file_pi:
        pickle.dump(model1.history, file_pi)