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

for k in range(1,11):
    for i in range(1, 10):
        K.clear_session()

        train_data = pd.read_csv('/data/internal/' + str(k) + '/data_train' + str(k) + str(i))
        train_val_data = pd.read_csv('/data/internal/' + str(k) + '/data_train_val' + str(k) + str(i))
        train_label = pd.read_csv('/data/internal/' + str(k) + '/label_train' + str(k) + str(i))
        train_val_label = pd.read_csv('/data/internal/' + str(k) + '/label_train_val' + str(k) + str(i))

        ##### If you need to add AF and Apnea data, the commented out code in this part may be useful #####

        # train_af_data = pd.read_csv('/data/wanghd/wanghong_s1/AF_data/' + str(k) + '/' + 'AF_train_' + str(k) + str(i) + '.csv')
        # val_af_data = pd.read_csv('/data/wanghd/wanghong_s1/AF_data/' + str(k) + '/' + 'AF_val_' + str(k) + str(i) + '.csv')
        # train_af_data = np.array(train_af_data)
        # val_af_data = np.array(val_af_data)
        # train_af_labels = np.zeros(shape=(train_af_data.shape[0], 1))
        # val_af_labels = np.zeros(shape=(val_af_data.shape[0], 1))
        #
        # train_apnea_data = pd.read_csv('/data/wanghd/wanghong_s1/Apnea_data/' + str(k) + '/' + 'train_train_' + str(k) + str(i) + '.csv')
        # val_apnea_data = pd.read_csv('/data/wanghd/wanghong_s1/Apnea_data/' + str(k) + '/' + 'train_val_' + str(k) + str(i) + '.csv')
        # train_apnea_data = np.array(train_apnea_data)
        # val_apnea_data = np.array(val_apnea_data)
        # train_apnea_labels = np.zeros(shape=(train_apnea_data.shape[0], 1))
        # val_apnea_labels = np.zeros(shape=(val_apnea_data.shape[0], 1))

        train_data = np.array(train_data)
        train_label = np.array(train_label)
        train_val_data = np.array(train_val_data)
        train_val_label = np.array(train_val_label)

        # train_count = count_label_number(train_label)
        # print('train_count:', train_count)
        # train_val_count = count_label_number(train_val_label)
        # print('train_val_label:', train_val_count)

        # train_data = np.concatenate([train_data, train_af_data, train_apnea_data], 0)
        # train_label = np.concatenate([train_label, train_af_labels, train_apnea_labels], 0)
        # train_val_data = np.concatenate([train_val_data, val_af_data, val_apnea_data], 0)
        # train_val_label = np.concatenate([train_val_label, val_af_labels, val_apnea_labels], 0)
        np.random.seed(2)
        np.random.shuffle(train_data)
        np.random.seed(2)
        np.random.shuffle(train_label)
        np.random.seed(2)
        np.random.shuffle(train_val_data)
        np.random.seed(2)
        np.random.shuffle(train_val_label)

        ########################################
        # train_ones_data, train_ones_labels = remake_data(1, train_count[1], train_data, train_label)
        # train_zeros_data, train_zeros_lables = remake_data(0, train_count[0], train_data, train_label)
        # val_ones_data, val_ones_labels = remake_data(1, train_val_count[1], train_val_data, train_val_label)
        # val_zeros_data, val_zeros_labels = remake_data(0, train_val_count[0], train_val_data, train_val_label)
        #
        # train_data = np.concatenate([train_ones_data, train_zeros_data], 0)
        # train_label = np.concatenate([train_ones_labels, train_zeros_lables], 0)
        # train_val_data = np.concatenate([val_ones_data, val_zeros_data], 0)
        # train_val_label = np.concatenate([val_ones_labels, val_zeros_labels], 0)
        #
        # np.random.seed(2)
        # np.random.shuffle(train_data)
        # np.random.seed(2)
        # np.random.shuffle(train_label)
        # np.random.seed(2)
        # np.random.shuffle(train_val_data)
        # np.random.seed(2)
        # np.random.shuffle(train_val_label)

        # train_data,val_data,train_label,val_label=train_test_split(train_val_data, train_val_label, test_size=0.15)
        train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
        train_val_data = train_val_data.reshape(train_val_data.shape[0], train_val_data.shape[1], 1)

        # train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)

        model1 = bulid_MultiScaleUnet2_Model()
        reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=5,mode = 'auto',factor=0.1)
        model1.compile(optimizer=optimizers.Adam(lr=1e-4), loss='categorical_hinge', metrics=['accuracy'])
        mc = ModelCheckpoint(filepath='cnn_ed_val2000' + str(k) + str(i) + '.h5', monitor='val_acc', mode='max',verbose=2, save_best_only=False)
        model1 = model1.fit(train_data, train_label, epochs=60, batch_size=16,validation_data=(train_val_data, train_val_label),verbose=2, callbacks=[mc, reduce_lr])

        validation_accuracy = model1.history['val_acc']
        df1 = pd.DataFrame(validation_accuracy)
        df1.to_csv('validation_acc' + str(k) + str(i), index=None)

        with open('trainHistoryDict' + str(k) + str(i) + '.txt', 'wb') as file_pi:
            pickle.dump(model1.history, file_pi)

