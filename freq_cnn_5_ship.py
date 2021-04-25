# -*- coding: utf-8 -*-
# Import Keras libraries and packages
from keras.models import Sequential  # Activate Neural Network
from keras.layers import Conv2D  # Convolution Operation
from keras.layers import MaxPooling2D # Pooling
from keras.layers import Flatten
from keras.layers import Dense # Fully Connected Networks
from keras.layers import Dropout
from keras.utils import np_utils # One Hot Encoding

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# 將數據調整成cnn網路所需的格式
def transfer_format_3x3(input_array):
    x0 = input_array[0]
    x1 = input_array[1]
    x2 = input_array[2]
    x3 = input_array[3]
    x4 = input_array[4]
    x5 = input_array[5]
    x6 = input_array[6]
    x7 = input_array[7]
    x8 = input_array[8]
  
    data = [x0,x1,x2,x3,x4,x5,x6,x7,x8]
    result = np.array(data).reshape(3,3)
    return result

def data_processing(file_path):
    read = pd.read_excel(file_path)
    read_data = read.iloc[1: ,: ].values
    
    get_data = []
    for i in range(read_data.shape[1]):
        data = transfer_format_3x3(read.iloc[1: , i].values)
        get_data.append(data)
    data_x = np.array(get_data)
    data_x_vector = data_x.reshape(len(data_x),3,3,1).astype('float32') # (21, 3, 3, 1)

    zeros = np.zeros( (read_data.shape[1]) )
    
    for i in range(len(zeros)):
        step = int(len(zeros)/5)
        if i < step:
            zeros[i] = 0
        elif step <= i < 2*step:
            zeros[i] = 1
        elif 2*step <= i < 3*step:
            zeros[i] = 2
        elif 3*step <= i < 4*step:
            zeros[i] = 3
        else:
            zeros[i] = 4

    data_label_onehot = np_utils.to_categorical(zeros)
    return data_x_vector, data_label_onehot

def result_graph(class_names , show_x , predictions , start , stop):
    show_y1 = []
    show_y2 = []
    show_y3 = []
    show_y4 = []
    show_y5 = []
    for i in range( start , stop ):
        show_y1.append(predictions[i][0])
        show_y2.append(predictions[i][1])
        show_y3.append(predictions[i][2])
        show_y4.append(predictions[i][3])
        show_y5.append(predictions[i][4])
    
    plt.figure()
    plt.plot(show_x, show_y1, 'r^-', label='with respect to type #1 (by CNN)')
    plt.plot(show_x, show_y2, 'bd-', label='with respect to type #2 (by CNN)')
    plt.plot(show_x, show_y3, 'gv-', label='with respect to type #3 (by CNN)')
    plt.plot(show_x, show_y4, 'yd-', label='with respect to type #4 (by CNN)')
    plt.plot(show_x, show_y5, 'kv-', label='with respect to type #5 (by CNN)')
    plt.xlabel('Elevation angle θ (degree)')
    plt.ylabel('Probability of prediction')
    plt.legend()
    
    ship_type = int(stop / len(show_x))
    img_name = f'ship_{ship_type}_cnn.png'
    plt.savefig(img_name)
    
    plt.show()
    return show_y1,show_y2,show_y3,show_y4,show_y5

if __name__ == '__main__':
    import time
    #import random 
    #random.seed( 40 ) # 使Neural Network訓練結果一致
    train_x_vector, train_label_onehot = data_processing("train_freq_5_ship_gan.xlsx")
    test_x_vector, test_label_onehot = data_processing("test_freq_5_ship.xlsx")
    
    model = Sequential()
    
    # The Input Layer :
    model.add(Conv2D(filters=10, kernel_size=(2, 2),
                     padding='same',
                     activation='relu',
                     input_shape=(3,3,1)))
    
    
    # The Hidden Layers :
    model.add(Conv2D(filters=10, kernel_size=(2, 2),
                     padding='same',
                     activation='relu'))
    
    model.add(MaxPooling2D(pool_size=(2, 2),strides=1))
    
    #model.add(Dropout(0.2))
    
    model.add(Flatten())#壓成一維
    
    model.add(Dense(128, activation='relu'))
    
    # The Output Layer :
    model.add(Dense(5, activation='softmax'))
    
    # Compile the network :
    model.compile(loss='categorical_crossentropy',optimizer= 'adam',metrics=['accuracy'],)
    model.summary()
    
    train_history = model.fit(train_x_vector, train_label_onehot, epochs = 100,verbose = 1)
    
    # 取得 struct_time 格式的時間
    t = time.localtime()
    # 依指定格式輸出
    result = time.strftime("-%m-%d-%H%M", t)
    
    output_model = 'freq_cnn_5_ship_gan'+result+'.h5'
    
    model.save(output_model)
    
    predictions = model.predict(test_x_vector)

    class_names = ['ship1','ship2','ship3','ship4','ship5']
    
    show_x = list(range(61,90,2))
    
    number_per_ship = int(predictions.shape[0] / 5)
    
    # ship1
    pred1_type1 ,pred2_type1 ,pred3_type1, pred4_type1 ,pred5_type1 = result_graph(class_names , show_x , predictions , 0 , number_per_ship)
    # ship2
    pred1_type2 ,pred2_type2 ,pred3_type2 ,pred4_type2 ,pred5_type2 = result_graph(class_names , show_x , predictions , number_per_ship , 2 * number_per_ship)
    # ship3
    pred1_type3 ,pred2_type3 ,pred3_type3 ,pred4_type3 ,pred5_type3 = result_graph(class_names , show_x , predictions , 2 * number_per_ship, 3 * number_per_ship)
    # ship4
    pred1_type4 ,pred2_type4 ,pred3_type4 ,pred4_type4 ,pred5_type4 = result_graph(class_names , show_x , predictions , 3 * number_per_ship, 4 * number_per_ship)
    # ship5
    pred1_type5 ,pred2_type5 ,pred3_type5 ,pred4_type5 ,pred5_type5 = result_graph(class_names , show_x , predictions , 4 * number_per_ship, 5 * number_per_ship)
    








