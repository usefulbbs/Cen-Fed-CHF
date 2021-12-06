import h5py
import pandas as pd
import numpy as np
file_train = h5py.File('/home/b227/PycharmProjects/H_Zexin/datasetFinal/dataset2/train_client1.h5','r')
file_test = pd.read_csv('/home/b227/PycharmProjects/H_Zexin/datasetFinal/dataset2/data_val1')
file_test = np.array(file_test)
length = file_test.shape[0]
print(length)
for i in range(2):
    file = file_train['examples']['client_'+str(i)]['pixels']
    print(file)
    length = length+file.shape[0]
print(length)