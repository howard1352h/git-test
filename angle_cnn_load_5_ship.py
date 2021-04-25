# -*- coding: utf-8 -*-
from keras.utils import np_utils # One Hot Encoding
from keras.models import load_model

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
    
    
    ship_type = int(stop / len(show_x))
    plt.legend(bbox_to_anchor=(0.2, 1),title = f'testing target: type #{ship_type}')
    img_name = f'ship_{ship_type}_cnn.png'
    
    
    plt.savefig(img_name, bbox_inches='tight')
    
    plt.show()
    return show_y1,show_y2,show_y3,show_y4,show_y5

if __name__ == '__main__':
    
    test_x_vector, test_label_onehot = data_processing('test_angle_5_ship.xlsx')
    model = load_model('angle_cnn_5_ship_gan.h5')
    
    model.summary()
    
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
    
    errors = 0
    for i in range(len(pred1_type1)):
        if pred1_type1[i] < pred2_type1[i] or pred1_type1[i] < pred3_type1[i] or pred1_type1[i] < pred4_type1[i] or pred1_type1[i] < pred5_type1[i]:
            errors += 1 
        if pred2_type2[i] < pred1_type2[i] or pred2_type2[i] < pred3_type2[i] or pred2_type2[i] < pred4_type2[i] or pred2_type2[i] < pred5_type2[i]:
            errors += 1 
        if pred3_type3[i] < pred1_type3[i] or pred3_type3[i] < pred2_type3[i] or pred3_type3[i] < pred4_type3[i] or pred3_type3[i] < pred5_type3[i]:
            errors += 1 
        if pred4_type4[i] < pred1_type4[i] or pred4_type4[i] < pred2_type4[i] or pred4_type4[i] < pred3_type4[i] or pred4_type4[i] < pred5_type4[i]:
            errors += 1 
        if pred5_type5[i] < pred1_type5[i] or pred5_type5[i] < pred2_type5[i] or pred5_type5[i] < pred3_type5[i] or pred5_type5[i] < pred4_type5[i]:
            errors += 1 
    print('error:',errors)
    print('total:', predictions.shape[0])
    