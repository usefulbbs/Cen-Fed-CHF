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
from sklearn.metrics import roc_curve, auc


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense,  BatchNormalization, GlobalAveragePooling1D,  concatenate, add, Reshape
from tensorflow.keras.layers import Conv1D, MaxPooling1D ,multiply,Lambda
from tensorflow.keras.layers import Activation

from tensorflow.python.keras import backend as K


NUM_EPOCHS = 1
BATCH_SIZE = 4
SHUFFLE_BUFFER = 50
PREFETCH_BUFFER = 4096
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# cpu_device = tf.config.list_logical_devices('CPU')
gpu_device = tf.config.list_logical_devices('GPU')
if not gpu_device:
    raise ValueError('not GPU')
# # tf.config.set_logical_device_configuration(
# #     gpu_device[0],
# #     [tf.config.LogicalDeviceConfiguration(memory_limit=8192),
# #      tf.config.LogicalDeviceConfiguration(memory_limit=8192)]
# # )
# # tf.config.list_logical_devices()
tff.backends.native.set_local_execution_context(client_tf_devices=gpu_device)

def map_fn(example):
  return collections.OrderedDict(
      x=tf.reshape(example['pixels'], [-1, 2000]),
      y=example['label'])  #example['label']


def client_data(n):
  ds = emnist_train.create_tf_dataset_for_client(n)
  return ds.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(BATCH_SIZE).map(map_fn)


def create_keras_model():
    def gelu(x):
        """基于Tanh近似计算的gelu函数
        """
        cdf = 0.5 * (1.0 + K.tanh(
            (np.sqrt(2 / np.pi) * (x + 0.044715 * K.pow(x, 3)))))
        return x * cdf

    def bilinear_upsampling(x, size):
        x = tf.expand_dims(x, axis=2)
        x = tf.image.resize(x, size=(x.shape[1] * size, 1))
        out = x[:, :, 0, :]
        return out
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
    def Conv_res(x,nb_fileter,kernerl_size):
        x = Conv1D(nb_fileter,kernerl_size,padding='same')(x)
        x = Lambda(gelu)(x)
        x0 = x
        x = BatchNormalization()(x,training=True)
        x = Conv1D(nb_fileter,kernerl_size,padding='same')(x)
        x = Lambda(gelu)(x)
        # x = Dropout(0.1)(x)
        x = BatchNormalization()(x,training=True)
        x = squeeze_excite_block(x, nb_fileter)
        x = add([x,x0])
        x = BatchNormalization()(x,training=True)
        return x

    input = Input(shape=(2000, 1), name='input')
    # x=ZeroPadding1D(8)(input)
    x = input

    encoder1 = Conv_res(x, 24, 3)
    pool1 = MaxPooling1D(2)(encoder1)

    encoder2 = Conv_res(pool1, 48, 3)
    pool2 = MaxPooling1D(2)(encoder2)

    encoder3 = Conv_res(pool2, 32, 3)
    pool3 = MaxPooling1D(2)(encoder3)

    encoder4 = Conv_res(pool3, 16, 3)
    pool4 = MaxPooling1D(2, name='pool4')(encoder4)

    encoder5 = Conv_res(pool4, 8, 3)
    # ===============================
    decoder5 = encoder5

    ####decoder4
    decoder4_4 = Conv1D(24, 3, padding='same')(encoder4)

    decoder4_5 = Lambda(bilinear_upsampling, arguments={'size': 2})(decoder5)
    decoder4_5 = Conv1D(24, 3, padding='same')(decoder4_5)

    decoder4_3 = MaxPooling1D(2)(encoder3)
    decoder4_3 = Conv1D(24, 3, padding='same')(decoder4_3)

    decoder4_2 = MaxPooling1D(4)(encoder2)
    decoder4_2 = Conv1D(24, 3, padding='same')(decoder4_2)

    decoder4_1 = MaxPooling1D(8)(encoder1)
    decoder4_1 = Conv1D(24, 3, padding='same')(decoder4_1)

    decoder4 = concatenate([decoder4_5, decoder4_4, decoder4_3, decoder4_2, decoder4_1], axis=2)
    decoder4 = Conv1D(120, 3, padding='same')(decoder4)
    decoder4 = Lambda(gelu)(decoder4)
    decoder4 = BatchNormalization()(decoder4,training=True)
    ####decoder3

    decoder3_3 = Conv1D(24, 3, padding='same')(encoder3)

    decoder3_5 = Lambda(bilinear_upsampling, arguments={'size': 4})(decoder5)
    decoder3_5 = Conv1D(24, 3, padding='same')(decoder3_5)

    decoder3_4 = Lambda(bilinear_upsampling, arguments={'size': 2})(decoder4)
    decoder3_4 = Conv1D(24, 3, padding='same')(decoder3_4)

    decoder3_2 = MaxPooling1D(2)(encoder2)
    decoder3_2 = Conv1D(24, 3, padding='same')(decoder3_2)

    decoder3_1 = MaxPooling1D(4)(encoder1)
    decoder3_1 = Conv1D(24, 3, padding='same')(decoder3_1)

    decoder3 = concatenate([decoder3_5, decoder3_4, decoder3_3, decoder3_2, decoder3_1], axis=2)
    decoder3 = Conv1D(120, 3, padding='same')(decoder3)
    decoder3 = Lambda(gelu)(decoder3)
    decoder3 = BatchNormalization()(decoder3,training=True)

    # decoder2
    decoder2_2 = encoder2
    decoder2_5 = Lambda(bilinear_upsampling, arguments={'size': 8})(decoder5)
    decoder2_5 = Conv1D(24, 3, padding='same')(decoder2_5)

    decoder2_4 = Lambda(bilinear_upsampling, arguments={'size': 4})(decoder4)
    decoder2_4 = Conv1D(24, 3, padding='same')(decoder2_4)

    decoder2_3 = Lambda(bilinear_upsampling, arguments={'size': 2})(decoder3)
    decoder2_3 = Conv1D(24, 3, padding='same')(decoder2_3)

    decoder2_1 = MaxPooling1D(2)(encoder1)
    decoder2_1 = Conv1D(24, 3, padding='same')(decoder2_1)

    decoder2 = concatenate([decoder2_5, decoder2_4, decoder2_3, decoder2_2, decoder2_1], axis=2)
    decoder2 = Conv1D(120, 3, padding='same')(decoder2)
    decoder2 = Lambda(gelu)(decoder2)
    decoder2 = BatchNormalization()(decoder2,training=True)

    # decoder1
    decoder1_1 = encoder1

    decoder1_5 = Lambda(bilinear_upsampling, arguments={'size': 16})(decoder5)
    decoder1_5 = Conv1D(24, 3, padding='same')(decoder1_5)

    decoder1_4 = Lambda(bilinear_upsampling, arguments={'size': 8})(decoder4)
    decoder1_4 = Conv1D(24, 3, padding='same')(decoder1_4)

    decoder1_3 = Lambda(bilinear_upsampling, arguments={'size': 4})(decoder3)
    decoder1_3 = Conv1D(24, 3, padding='same')(decoder1_3)

    decoder1_2 = Lambda(bilinear_upsampling, arguments={'size': 2})(decoder2)
    decoder1_2 = Conv1D(24, 3, padding='same')(decoder1_2)

    decoder1 = concatenate([decoder1_5, decoder1_4, decoder1_3, decoder1_2, decoder1_1], axis=2)
    decoder1 = Conv1D(120, 3, padding='same')(decoder1)
    decoder1 = Lambda(gelu)(decoder1)
    decoder1 = BatchNormalization()(decoder1,training=True)

    def last_step(input, name=None):
        outputx = GlobalAveragePooling1D()(input)
        outputx = Dense(128)(outputx)
        outputx = Activation('sigmoid')(outputx)
        outputx = Dense(1)(outputx)
        outputx = Activation('sigmoid', name=name)(outputx)
        return outputx

    output1 = last_step(decoder1, name='output1')
    output2 = last_step(decoder2, name='output2')
    output3 = last_step(decoder3, name='output3')
    output4 = last_step(decoder4, name='output4')
    output5 = last_step(decoder5, name='output5')
    weight_1 = Lambda(lambda x: x * 0.4)
    weight_2 = Lambda(lambda x: x * 0.2)
    weight_3 = Lambda(lambda x: x * 0.2)
    weight_4 = Lambda(lambda x: x * 0.1)
    weight_5 = Lambda(lambda x: x * 0.1)
    output1 = weight_1(output1)
    output2 = weight_2(output2)
    output3 = weight_3(output3)
    output4 = weight_4(output4)
    output5 = weight_5(output5)
    outputs = add([output5,output4,output3,output2,output1])
    # model1 = Model(inputs=input,outputs=[output11,output22,output33,output44])
    model1 = Model(inputs=input, outputs=outputs)
    return model1

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
NUM_CLIENTS = 4
num_rounds = 230
for i in [NUM_CLIENTS]:
    client_num = i
    path = r'/home/b227/PycharmProjects/H_Zexin/tff_dataset/dataset4'
    print(path)
    for j in [4]:
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