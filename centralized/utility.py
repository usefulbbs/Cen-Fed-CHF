import numpy as np

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