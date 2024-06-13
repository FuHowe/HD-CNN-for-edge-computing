import torch
import mypath
import scipy.io as scio
import os
import copy
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from mymodel import myCNN_Train
from mymodel import myCNN_Para
from mymodel import myDATA
import shutil
import mycnn
import time
import myfile
import scipy.signal as signal
import matplotlib.pyplot as plt
from PIL import Image
from torchsummary import summary
from torchvision import transforms
from torch.nn import functional
import math
from sklearn.preprocessing import MinMaxScaler
import struct


def lw_init():
    
    for file in os.listdir(os.getcwd()):
        if file.endswith('.xlsx'):
            os.remove(file)

    
    current_path = os.getcwd()
    os.chdir(mypath.model_path)
    for file in os.listdir(os.path.join(current_path, mypath.model_path)):
        if file.endswith('.bin'):
            os.remove(file)
    os.chdir(current_path)

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device



def lw_get_full_connect_len(image_size, input_dim_1, output_dim_1, kernel_1, pooling_1, input_dim_2, output_dim_2,kernel_2, pooling_2, input_dim_3, output_dim_3, kernel_3, pooling_3, classification_out):
    img = torch.rand(1, 1, image_size, image_size)  

    model = myCNN_Para(input_dim_1, output_dim_1, kernel_1, pooling_1, input_dim_2, output_dim_2, kernel_2, pooling_2,input_dim_3, output_dim_3, kernel_3, pooling_3, classification_out)  

    length = model(img).detach().numpy().shape[1]
    return length



def lw_parameter_volume(image_size, input_dim_1, output_dim_1, kernel_1, pooling_1, input_dim_2, output_dim_2, kernel_2,pooling_2, input_dim_3, output_dim_3, kernel_3, pooling_3, classification_out):
    bytes_num = 4  
    kbytes = 1024  
    head_num = 18  

    
    

    
    full_connect_len = mycnn.lw_get_full_connect_len(image_size, input_dim_1, output_dim_1, kernel_1, pooling_1,input_dim_2, output_dim_2, kernel_2, pooling_2, input_dim_3,output_dim_3, kernel_3, pooling_3, classification_out)  

    
    head_len = head_num * bytes_num / kbytes

    
    conv_1 = (output_dim_1 * input_dim_1 * kernel_1 * kernel_1 + output_dim_1) * bytes_num / kbytes
    conv_2 = (output_dim_2 * output_dim_1 * kernel_2 * kernel_2 + output_dim_2) * bytes_num / kbytes
    conv_3 = (output_dim_3 * output_dim_2 * kernel_3 * kernel_3 + output_dim_3) * bytes_num / kbytes
    linear_1 = (full_connect_len * classification_out + classification_out) * bytes_num / kbytes

    
    img_size = (image_size * image_size) * bytes_num / kbytes

    
    para_size = (conv_1 + conv_2 + conv_3 + linear_1) * kbytes / bytes_num  

    
    total_size = head_len + conv_1 + conv_2 + conv_3 + linear_1 + img_size

    return head_len, conv_1, conv_2, conv_3, linear_1, img_size, para_size, total_size



def lw_cnn_parameter_check(inputsize, kernel_1, pooling_1, kernel_2, pooling_2, kernel_3, pooling_3):
    
    conv_1_h = math.floor((inputsize + 2 * 0 - 1 * (kernel_1 - 1) - 1) / 1 + 1)  
    pooling_1_h = math.floor((conv_1_h + 2 * 0 - 1 * (pooling_1 - 1) - 1) / pooling_1 + 1)  

    
    conv_2_h = math.floor((pooling_1_h + 2 * 0 - 1 * (kernel_2 - 1) - 1) / 1 + 1)  
    pooling_2_h = math.floor((conv_2_h + 2 * 0 - 1 * (pooling_2 - 1) - 1) / pooling_2 + 1)  

    
    conv_3_h = math.floor((pooling_2_h + 2 * 0 - 1 * (kernel_3 - 1) - 1) / 1 + 1)  
    

    if ((inputsize < kernel_1) or (conv_1_h < pooling_1) or (pooling_1_h < kernel_2) or \
            (conv_2_h < pooling_2) or (pooling_2_h < kernel_3) or (conv_3_h < pooling_3)):
        return -1
    else:
        return 1



def lw_parameter_check(image_size, input_dim_1, output_dim_1, kernel_1, pooling_1, input_dim_2, output_dim_2, kernel_2,pooling_2, input_dim_3, output_dim_3, kernel_3, pooling_3, classification_out):
    
    

    STM32_TOTAL_MALLOC_SIZE = 128 * 1  
    STM32_TEST_FLASH_SIZE = 960 * 1  

    
    if -1 == lw_cnn_parameter_check(image_size, kernel_1, pooling_1, kernel_2, pooling_2, kernel_3, pooling_3):
        return -1
    else:
        return 1

    
    head_len, conv_1, conv_2, conv_3, linear_1, img_size, para_size, total_size = lw_parameter_volume(image_size,input_dim_1,output_dim_1,kernel_1,pooling_1,input_dim_2,output_dim_2,kernel_2,pooling_2,input_dim_3,output_dim_3,kernel_3,pooling_3,classification_out)

    
    if (((conv_1 - head_len) > STM32_TOTAL_MALLOC_SIZE) or
            ((conv_2 - head_len) > STM32_TOTAL_MALLOC_SIZE) or
            ((conv_3 - head_len) > STM32_TOTAL_MALLOC_SIZE) or
            ((linear_1 - head_len) > STM32_TOTAL_MALLOC_SIZE) or
            (total_size > STM32_TEST_FLASH_SIZE)):
        return -1
    else:
        return 1



def lw_model_out(test_original_data, model_para):
    
    window_len = np.uint32(2 * model_para.image_size)
    shift_len = np.uint32(window_len / 2)
    test_image = lw_stft_fft(len(test_original_data), window_len, shift_len, test_original_data)

    
    model_out_directly_stft = mycnn.lw_model_out_directly(copy.deepcopy(test_image))

    
    model_out_directly = mycnn.lw_model_out_directly(copy.deepcopy(test_image))

    
    model_out_hierarchy = mycnn.lw_model_out_hierarchy(copy.deepcopy(test_image), copy.deepcopy(model_para))

    
    model_out_layer = mycnn.lw_model_out_layer(copy.deepcopy(test_image), copy.deepcopy(model_para))

    
    model_out_detial = mycnn.lw_model_out_detial(copy.deepcopy(test_image), copy.deepcopy(model_para))

    return model_out_directly_stft



def lw_edge_computing(device, single, count, choice, image_size, data_len, batch_size, learning_rate, epochs, input_dim_1,output_dim_1, kernel_1, pooling_1, input_dim_2, output_dim_2, kernel_2, pooling_2, input_dim_3,output_dim_3, kernel_3, pooling_3, classification_out):
    
    if -1 == lw_parameter_check(image_size, input_dim_1, output_dim_1, kernel_1, pooling_1, input_dim_2, output_dim_2,kernel_2, pooling_2, input_dim_3, output_dim_3, kernel_3, pooling_3,classification_out):
        return -1
    else:
        
        
        full_connect_len = mycnn.lw_get_full_connect_len(image_size, input_dim_1, output_dim_1, kernel_1, pooling_1, input_dim_2, output_dim_2, kernel_2, pooling_2, input_dim_3,output_dim_3, kernel_3, pooling_3, classification_out)  

        
        if choice == 'CWRU':
            train_dataloader, test_dataloader, normal_sample = mycnn.lw_rnn_CWRU_set(single, image_size, batch_size, data_len)
        elif choice == 'NTU':
            train_dataloader, test_dataloader, normal_sample = mycnn.lw_rnn_NTU_set(single, image_size, batch_size, data_len)
        elif choice == 'CQU_1':
            train_dataloader, test_dataloader, normal_sample = mycnn.lw_rnn_CQU_1_set(single, image_size, batch_size, data_len)
        elif choice == 'CQU_2':
            train_dataloader, test_dataloader, normal_sample = mycnn.lw_rnn_CQU_2_set(single, image_size, batch_size, data_len)

        
        model = myCNN_Train(input_dim_1, output_dim_1, kernel_1, pooling_1, input_dim_2, output_dim_2, kernel_2, pooling_2, input_dim_3, output_dim_3, kernel_3, pooling_3, full_connect_len,classification_out).to(device)  

        
        if single == 1:
            summary(model, (1, image_size, image_size), )  

        
        loss_fn, optimizer = mycnn.lw_dnn_init(device, learning_rate, model)

        
        best_correct_rate, use_time = mycnn.lw_dnn_train_test(single, device, train_dataloader, test_dataloader, model,loss_fn, optimizer, epochs)

        
        head_len, conv_1, conv_2, conv_3, linear_1, img_size, para_size, total_size = lw_parameter_volume(image_size,input_dim_1,output_dim_1,kernel_1, pooling_1,input_dim_2,output_dim_2,kernel_2,pooling_2,input_dim_3,output_dim_3,kernel_3,pooling_3,classification_out)

        
        model_para = mycnn.lw_split_model(para_size, image_size, input_dim_1, output_dim_1, kernel_1, pooling_1,input_dim_2, output_dim_2, kernel_2, pooling_2, input_dim_3, output_dim_3,kernel_3, pooling_3, full_connect_len, classification_out)

        
        max_ram = mycnn.lw_computing_max_ram(model_para)

        
        test_original_data, file_name = mycnn.lw_save_parameters(single, count, para_size, copy.deepcopy(model_para), copy.deepcopy(normal_sample))

        
        if single == 1:
            model_out_directly_stft = lw_model_out(copy.deepcopy(test_original_data),copy.deepcopy(model_para))

        
        myfile.dt_export_result(count, best_correct_rate, image_size, input_dim_1, output_dim_1, kernel_1, pooling_1,input_dim_2, output_dim_2, kernel_2, pooling_2, input_dim_3, output_dim_3, kernel_3,pooling_3, full_connect_len, classification_out, head_len, conv_1, conv_2, conv_3,linear_1, img_size, total_size, max_ram)

        print('T=%8d, Ac=%-0.2f, Hl=%-5.2f K, C1=%-5.2f K, C2=%-5.2f K, C3=%-5.2f K, L1=%-5.2f K, Img=%-5.2f K, Ts=%-5.2f K, Ut=%-5.2f s'% (count, best_correct_rate, head_len, conv_1, conv_2, conv_3, linear_1, img_size, total_size, use_time))

        
        if single == 1:
            myfile.lw_download_data(file_name)
            out = ['{:.6f}'.format(out) for out in model_out_directly_stft]
            out_1 = [float(num) for num in out]
            print('model_out_directly_stft = model_out:{}'.format(out_1))
            return model_para, model_para.weight_bias



def lw_train_data(image_size, indata):
    
    window_len = image_size * 2  
    shift_len = window_len / 2  
    one_img_stft_len =  (image_size -1) *shift_len + window_len  

    train_data = []
    step_len = 32
    for i in range(0, np.uint32((len(indata) - one_img_stft_len) / step_len)):
        train_data.append(indata[np.uint32(step_len * i) : np.uint32(step_len * i + one_img_stft_len)])

    return train_data



def lw_stft_set(image_size, indata):
    
    window_len = 2 * image_size 
    shift_len = window_len/2  

    imgs_arr = []
    for i in range(0, len(indata)):
        img = lw_stft_fft(len(indata[i]), window_len, shift_len, indata[0])
        img = torch.Tensor(img).unsqueeze(0) 

        
        
        

        imgs_arr.append(img)

    return imgs_arr



def lw_stft_fft(input_len, window_len, shift_len, stft_input):
    sample_frequency = 12000  

    
    hamming_window = []
    for i in range(window_len):
        hamming_window.append(np.float32(0.5 - 0.5 * np.cos(2 * np.pi * i / (window_len - 1))))

    
    stft_out_1 = []
    for i in range(np.uint32((np.uint32(input_len - window_len) / shift_len + 1))):
        window_data = stft_input[int(i * shift_len): int(i * shift_len + window_len)]

        
        hamming_data = []
        for j in range(window_len):
            hamming_data.append(np.float32(window_data[j]) * hamming_window[j])

        hamming_data = np.array(hamming_data).reshape(-1, 1)

        
        fft_data_1 = np.fft.fft2(hamming_data)
        fft_data_1_normalized = fft_data_1.flatten() / (window_len / 2)  
        fft_data_1_oneside = fft_data_1_normalized[0: int((window_len / 2))]  

        fft_data_1_abs = np.abs(fft_data_1_oneside)  
        stft_out_1.append(np.float32(fft_data_1_abs))

    stft_out_1 = np.array(stft_out_1)

    return stft_out_1



def lw_stft_dft(input_len, window_len, shift_len, stft_input):
    sample_frequency = 12000  
    
    hamming_window = []
    for i in range(window_len):
        hamming_window.append(0.5 - 0.5 * np.cos(2 * np.pi * i / (window_len - 1)))

    
    stft_out_3 = []
    for i in range(np.uint32((np.uint32(input_len - window_len) / shift_len + 1))):
        window_data = stft_input[int(i * shift_len): int(i * shift_len + window_len)]

        
        hamming_data = []
        for j in range(window_len):
            hamming_data.append(window_data[j] * hamming_window[j])

        hamming_data = np.array(hamming_data).reshape(-1, 1)

        
        
        fft_data_3 = []
        for k in range(np.uint32(window_len)):  
            real = 0  
            imag = 0  
            for n in range(window_len):
                real += (hamming_data[n] * np.cos(2 * np.pi * k * n / window_len))
                imag += (hamming_data[n] * np.sin(2 * np.pi * k * n / window_len))

            fft_data_3.append(np.float32(np.sqrt(real * real + imag * imag)))

        fft_data_3_normalized = np.array(fft_data_3).flatten() / (window_len / 2)  
        fft_data_3_oneside = fft_data_3_normalized[0: int((window_len / 2))]  

        stft_out_3.append(fft_data_3_oneside)

    stft_out_3 = np.array(stft_out_3)

    return stft_out_3


def lw_rnn_CQU_1_set(single, image_size, batch_size, data_len):
    train_ratio = 0.8  
    point_bytes = 4  
    file_path_1 = r'E:\1_Data\5_����\12_��������\2_��DDS̨��(�������)\DDS���ݿ�ȫ\DDS���Թ��Ͽ�\DATA\ƽ�г�������й�������Ϻ���\DATA'
    file_path_2 = r'E:\1_Data\5_����\12_��������\2_��DDS̨��(�������)\DDS���ݿ�ȫ\DDS���Թ��Ͽ�\DATA\ƽ�г����������Ȧ���Ϻ���\DATA'
    file_path_3 = r'E:\1_Data\5_����\12_��������\2_��DDS̨��(�������)\DDS���ݿ�ȫ\DDS���Թ��Ͽ�\DATA\ƽ�г����������Ȧ���Ϻ���\DATA'
    file_path_4 = r'E:\1_Data\5_����\12_��������\2_��DDS̨��(�������)\DDS���ݿ�ȫ\DDS���Թ��Ͽ�\DATA\ƽ�г�������и��Ϲ��Ϻ���\DATA'
    file_name = 'dds���Թ��Ͽ�4.6

    
    os.chdir(file_path_1)
    with open(file_name, 'rb') as file:
        normal_data = np.float32(struct.unpack('f' * data_len, file.read(point_bytes * data_len)))  
    
    os.chdir(file_path_2)
    with open(file_name, 'rb') as file:
        inner_data = np.float32(struct.unpack('f' * data_len, file.read(point_bytes * data_len)))
    
    os.chdir(file_path_3)
    with open(file_name, 'rb') as file:
        outer_data = np.float32(struct.unpack('f' * data_len, file.read(point_bytes * data_len)))
    
    os.chdir(file_path_4)
    with open(file_name, 'rb') as file:
        complex_data = np.float32(struct.unpack('f' * data_len, file.read(point_bytes * data_len)))

    os.chdir(mypath.project_path)

    
    data_normal = MinMaxScaler(feature_range=(0, 1)).fit_transform(normal_data.reshape(-1, 1))  
    data_inner = MinMaxScaler(feature_range=(0, 1)).fit_transform(inner_data.reshape(-1, 1))  
    data_complex = MinMaxScaler(feature_range=(0, 1)).fit_transform(complex_data.reshape(-1, 1))  
    data_outer = MinMaxScaler(feature_range=(0, 1)).fit_transform(outer_data.reshape(-1, 1))  

    
    normal_sample = lw_train_data(image_size, copy.deepcopy(data_normal))
    inner_sample = lw_train_data(image_size, copy.deepcopy(data_inner))
    complex_sample = lw_train_data(image_size, copy.deepcopy(data_complex))
    outer_sample = lw_train_data(image_size, copy.deepcopy(data_outer))

    
    normal_imgs = lw_stft_set(image_size, copy.deepcopy(normal_sample))
    inner_imgs = lw_stft_set(image_size, copy.deepcopy(inner_sample))
    complex_imgs = lw_stft_set(image_size, copy.deepcopy(complex_sample))
    outer_imgs = lw_stft_set(image_size, copy.deepcopy(outer_sample))



    merge_imgs = []
    merge_imgs.extend(normal_imgs)
    merge_imgs.extend(inner_imgs)
    merge_imgs.extend(complex_imgs)
    merge_imgs.extend(outer_imgs)
    merge_imgs = torch.stack(merge_imgs)

    
    normal_label = torch.full((len(normal_imgs), 1), 0).flatten()
    inner_label = torch.full((len(inner_imgs), 1), 1).flatten()
    complex_label = torch.full((len(complex_imgs), 1), 2).flatten()
    outer_label = torch.full((len(outer_imgs), 1), 3).flatten()

    
    merge_label = []
    merge_label.extend(normal_label)
    merge_label.extend(inner_label)
    merge_label.extend(complex_label)
    merge_label.extend(outer_label)
    merge_label = torch.stack(merge_label)

    
    merge_set = TensorDataset(merge_imgs, merge_label)  

    
    train_set, test_set = torch.utils.data.random_split(merge_set, [np.uint32(len(merge_set) * train_ratio),len(merge_set) - np.uint32(len(merge_set) * train_ratio)])

    
    train_dataloader = DataLoader(train_set, batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_dataloader = DataLoader(test_set, batch_size, shuffle=True, num_workers=0, drop_last=True)


    return train_dataloader, test_dataloader, normal_sample


def lw_rnn_CQU_2_set(single, image_size, batch_size, data_len):
    train_ratio = 0.8  
    point_bytes = 4  
    file_path_1 = r'E:\1_Data\5_����\12_��������\2_��DDS̨��(�������)\DDS���ݿ�ȫ\DDS���Թ��Ͽ�\DATA\���ǳ����������������\DATA'
    file_path_2 = r'E:\1_Data\5_����\12_��������\2_��DDS̨��(�������)\DDS���ݿ�ȫ\DDS���Թ��Ͽ�\DATA\���ǳ����������Ȧ���Ϻ���\DATA'
    file_path_3 = r'E:\1_Data\5_����\12_��������\2_��DDS̨��(�������)\DDS���ݿ�ȫ\DDS���Թ��Ͽ�\DATA\���ǳ����������Ȧ���Ϻ���\DATA'
    file_path_4 = r'E:\1_Data\5_����\12_��������\2_��DDS̨��(�������)\DDS���ݿ�ȫ\DDS���Թ��Ͽ�\DATA\���ǳ����������Ϲ��Ϻ���\DATA'

    file_name = 'dds���Թ��Ͽ�4.6

    
    os.chdir(file_path_1)
    with open(file_name, 'rb') as file:
        normal_data = np.float32(struct.unpack('f' * data_len, file.read(point_bytes * data_len)))  
    
    os.chdir(file_path_2)
    with open(file_name, 'rb') as file:
        inner_data = np.float32(struct.unpack('f' * data_len, file.read(point_bytes * data_len)))
    
    os.chdir(file_path_3)
    with open(file_name, 'rb') as file:
        outer_data = np.float32(struct.unpack('f' * data_len, file.read(point_bytes * data_len)))
    
    os.chdir(file_path_4)
    with open(file_name, 'rb') as file:
        complex_data = np.float32(struct.unpack('f' * data_len, file.read(point_bytes * data_len)))

    os.chdir(mypath.project_path)

    
    data_normal = MinMaxScaler(feature_range=(0, 1)).fit_transform(normal_data.reshape(-1, 1))  
    data_inner = MinMaxScaler(feature_range=(0, 1)).fit_transform(inner_data.reshape(-1, 1))  
    data_complex = MinMaxScaler(feature_range=(0, 1)).fit_transform(complex_data.reshape(-1, 1))  
    data_outer = MinMaxScaler(feature_range=(0, 1)).fit_transform(outer_data.reshape(-1, 1))  

    
    normal_sample = lw_train_data(image_size, copy.deepcopy(data_normal))
    inner_sample = lw_train_data(image_size, copy.deepcopy(data_inner))
    complex_sample = lw_train_data(image_size, copy.deepcopy(data_complex))
    outer_sample = lw_train_data(image_size, copy.deepcopy(data_outer))

    
    normal_imgs = lw_stft_set(image_size, copy.deepcopy(normal_sample))
    inner_imgs = lw_stft_set(image_size, copy.deepcopy(inner_sample))
    complex_imgs = lw_stft_set(image_size, copy.deepcopy(complex_sample))
    outer_imgs = lw_stft_set(image_size, copy.deepcopy(outer_sample))

    
    merge_imgs = []
    merge_imgs.extend(normal_imgs)
    merge_imgs.extend(inner_imgs)
    merge_imgs.extend(complex_imgs)
    merge_imgs.extend(outer_imgs)
    merge_imgs = torch.stack(merge_imgs)

    
    normal_label = torch.full((len(normal_imgs), 1), 0).flatten()
    inner_label = torch.full((len(inner_imgs), 1), 1).flatten()
    complex_label = torch.full((len(complex_imgs), 1), 2).flatten()
    outer_label = torch.full((len(outer_imgs), 1), 3).flatten()

    
    merge_label = []
    merge_label.extend(normal_label)
    merge_label.extend(inner_label)
    merge_label.extend(complex_label)
    merge_label.extend(outer_label)
    merge_label = torch.stack(merge_label)

    
    merge_set = TensorDataset(merge_imgs, merge_label)  

    
    train_set, test_set = torch.utils.data.random_split(merge_set, [np.uint32(len(merge_set) * train_ratio),len(merge_set) - np.uint32(len(merge_set) * train_ratio)])

    
    train_dataloader = DataLoader(train_set, batch_size, shuffle=True, num_workers=0, drop_last=True)
    test_dataloader = DataLoader(test_set, batch_size, shuffle=True, num_workers=0, drop_last=True)


    return train_dataloader, test_dataloader, normal_sample



















































































































































































def lw_rnn_NTU_set(single, image_size, batch_size, data_len):
    data_label = scio.loadmat(os.path.join(mypath.cnn_ntu_data_path, 'mylabel'))['mylabel'] 

    
    folders_num = 31
    files_num = 24
    sensor_num = 38

    folders_name = []
    files_name = []
    for i in range(folders_num):
        for j in range(files_num):
            for k in range(sensor_num):
                folder_name = '2012-01-' + str(i+1).zfill(2)
                file_name = folder_name  + ' ' + str(j).zfill(2) + '-VIB'

                folders_name.append(folder_name)
                files_name.append(file_name)


    
    normal_data = [] 
    missing_data = []  
    minor_data = [] 
    outlier_data = [] 
    square_data = [] 
    trend_data = [] 
    drift_data = []  

    normal_data_count = 0 
    missing_data_count = 0  
    minor_data_count = 0 
    outlier_data_count= 0 
    square_data_count = 0 
    trend_data_count = 0 
    drift_data_count = 0  

    row = 31*24 
    col = 38 

    each_sensor_num = 1 
    file_count = 0
    for i in range(row):
        for j in range(col):
            if data_label[i][j] != 8:
                path = os.path.join(os.path.join(mypath.ntu_all_data_path, folders_name[file_count]), files_name[file_count]) 
                if (data_label[i][j] == 1) and (normal_data_count < each_sensor_num):
                    read_data = scio.loadmat(path).get('data')[:, j][10000 : 10000 + data_len]  
                    normal_data.append(read_data) 
                    normal_data_count += 1

                if (data_label[i][j] == 2) and (missing_data_count < each_sensor_num):
                    read_data = np.zeros(data_len)[0: data_len] 
                    missing_data.append(read_data) 
                    missing_data_count += 1

                if (data_label[i][j] == 3) and (minor_data_count < each_sensor_num):
                    read_data = scio.loadmat(path).get('data')[:, j][0: data_len]
                    minor_data.append(read_data) 
                    minor_data_count += 1

                if (data_label[i][j] == 4) and (outlier_data_count < each_sensor_num):
                    read_data = scio.loadmat(path).get('data')[:, j][0: data_len]
                    outlier_data.append(read_data) 
                    outlier_data_count += 1

                if (data_label[i][j] == 5) and (square_data_count < each_sensor_num):
                    read_data = scio.loadmat(path).get('data')[:, j][0: data_len]
                    square_data.append(read_data) 
                    square_data_count += 1

                if (data_label[i][j] == 6) and (trend_data_count < each_sensor_num):
                    read_data = scio.loadmat(path).get('data')[:, j][0: data_len]
                    trend_data.append(read_data) 
                    trend_data_count += 1

                if (data_label[i][j] == 7) and (drift_data_count < each_sensor_num):
                    read_data = scio.loadmat(path).get('data')[:, j][0: data_len]
                    drift_data.append(read_data) 
                    drift_data_count += 1
            file_count += 1



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    data_normal = []
    data_missing = []
    data_minor = []
    data_outlier = []
    data_square = []
    data_trend = []
    data_drift = []
    for i in range(each_sensor_num):
        data_normal.append(MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array(normal_data[i]).reshape(-1, 1)))  
        data_missing.append(MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array(missing_data[i]).reshape(-1, 1)))  
        data_minor.append(MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array(minor_data[i]).reshape(-1, 1)))  
        data_outlier.append(MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array(outlier_data[i]).reshape(-1, 1)))  
        data_square.append(MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array(square_data[i]).reshape(-1, 1)))  
        data_trend.append(MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array(trend_data[i]).reshape(-1, 1)))  
        data_drift.append(MinMaxScaler(feature_range=(0, 1)).fit_transform(np.array(drift_data[i]).reshape(-1, 1)))  


    
    
    normal_sample = []
    missing_sample = []
    minor_sample = []
    outlier_sample = []
    square_sample  = []
    trend_sample = []
    drift_sample = []
    for i in range(each_sensor_num):
        normal_sample.extend(lw_train_data(image_size, copy.deepcopy(data_normal[i])))
        missing_sample.extend(lw_train_data(image_size, copy.deepcopy(data_missing[i])))
        minor_sample.extend(lw_train_data(image_size, copy.deepcopy(data_minor[i])))
        outlier_sample.extend(lw_train_data(image_size, copy.deepcopy(data_outlier[i])))
        square_sample.extend(lw_train_data(image_size, copy.deepcopy(data_square[i])))
        trend_sample.extend(lw_train_data(image_size, copy.deepcopy(data_trend[i])))
        drift_sample.extend(lw_train_data(image_size, copy.deepcopy(data_drift[i])))


    
    normal_imgs = lw_stft_set(image_size, copy.deepcopy(normal_sample))
    missing_imgs = lw_stft_set(image_size, copy.deepcopy(missing_sample))
    minor_imgs = lw_stft_set(image_size, copy.deepcopy(minor_sample))
    outlier_imgs = lw_stft_set(image_size, copy.deepcopy(outlier_sample))
    square_imgs = lw_stft_set(image_size, copy.deepcopy(square_sample))
    trend_imgs = lw_stft_set(image_size, copy.deepcopy(trend_sample))
    drift_imgs = lw_stft_set(image_size, copy.deepcopy(drift_sample))


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


    
    merge_sample = []
    merge_sample.extend(normal_imgs)
    merge_sample.extend(missing_imgs)
    merge_sample.extend(minor_imgs)
    merge_sample.extend(outlier_imgs)
    merge_sample.extend(square_imgs)
    merge_sample.extend(trend_imgs)
    merge_sample.extend(drift_imgs)
    merge_sample = torch.stack(merge_sample)

    
    normal_label = torch.full((len(normal_imgs), 1), 0).flatten()
    missing_label = torch.full((len(missing_imgs), 1), 1).flatten()
    minor_label = torch.full((len(minor_imgs), 1), 2).flatten()
    outlier_label = torch.full((len(outlier_imgs), 1), 3).flatten()
    square_label = torch.full((len(square_imgs), 1), 4).flatten()
    trend_label = torch.full((len(trend_imgs), 1), 5).flatten()
    drift_label = torch.full((len(drift_imgs), 1), 6).flatten()

    merge_label = []
    merge_label.extend(normal_label)
    merge_label.extend(missing_label)
    merge_label.extend(minor_label)
    merge_label.extend(outlier_label)
    merge_label.extend(square_label)
    merge_label.extend(trend_label)
    merge_label.extend(drift_label)
    merge_label = torch.stack(merge_label)

    
    merge_set = TensorDataset(merge_sample, merge_label)

    
    train_ratio = 0.8  
    train_set, test_set = torch.utils.data.random_split(merge_set, [np.uint32(len(merge_set) * train_ratio), len(merge_set) - np.uint32(len(merge_set) * train_ratio)])
    if single == 1:
        print("ѵ��������:%d, ���Լ�����:%d" % (len(train_set), len(test_set)))

    
    train_dataloader = DataLoader(train_set, batch_size, shuffle = True, num_workers = 0, drop_last = True)
    test_dataloader = DataLoader(test_set, batch_size, shuffle = True, num_workers = 0, drop_last = True)

    return train_dataloader, test_dataloader, normal_sample




def lw_dnn_init(device, learning_rate, model):
    loss_fn = nn.CrossEntropyLoss().to(device)  
    
    optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    
    
    return loss_fn, optimizer



def lw_dnn_train(device, train_dataloader, model, loss_fn, optimizer):
    model.train()  

    
    for imgs, labels in train_dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        pred = model(imgs)  

        loss = loss_fn(pred, labels)  
        optimizer.zero_grad()  
        loss.backward()
        optimizer.step()  



def lw_dnn_test(device, test_dataloader, model):
    model.eval()  

    correct_nums = 0  
    sub_epoch = 0  

    with torch.no_grad():  
        for imgs, labels in test_dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            pred = model(imgs)

            correct_nums += (pred.argmax(1) == labels).sum().item()
            sub_epoch += len(imgs)  

    correct_rate = (correct_nums / sub_epoch) * 100  

    return correct_rate



def lw_dnn_train_test(single, device, train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs):
    begin_time = time.time()

    final_correct_rate = -np.inf  

    for i in range(epochs):
        lw_dnn_train(device, train_dataloader, model, loss_fn, optimizer)  
        current_correct_rate = lw_dnn_test(device, test_dataloader, model)  

        if single == 1:
            print("Epoch:%d  Accuracy:%f " % (i + 1, current_correct_rate))

        
        if current_correct_rate > final_correct_rate:  
            final_correct_rate = current_correct_rate

            best_model = model
            torch.save(best_model, 'Best_CNN_model.pth')  

        if np.uint32(final_correct_rate) == 100:  
            break

    end_time = time.time()
    use_time = end_time - begin_time

    return final_correct_rate, use_time



def lw_split_model(para_size, image_size, input_dim_1, output_dim_1, kernel_1, pooling_1, input_dim_2, output_dim_2,kernel_2, pooling_2, input_dim_3, output_dim_3, kernel_3, pooling_3, full_connect_len,classification_out):
    model = torch.load('Best_CNN_model.pth')  
    model_dict = model.state_dict()  

    
    model_data = myDATA()
    model_data.head_len = np.float32(18)  
    model_data.con_layers = np.float32(4)  
    model_data.para_size = np.float32(para_size)  
    model_data.image_size = np.float32(image_size)  

    model_data.con_layer_input_dimension_1 = np.float32(input_dim_1)  
    model_data.con_layer_output_dimension_1 = np.float32(output_dim_1)  
    model_data.con_layer_kernel_1 = np.float32(kernel_1)  
    model_data.con_layer_pooling_1 = np.float32(pooling_1)  

    model_data.con_layer_input_dimension_2 = np.float32(input_dim_2)  
    model_data.con_layer_output_dimension_2 = np.float32(output_dim_2)  
    model_data.con_layer_kernel_2 = np.float32(kernel_2)  
    model_data.con_layer_pooling_2 = np.float32(pooling_2)  

    model_data.con_layer_input_dimension_3 = np.float32(input_dim_3)  
    model_data.con_layer_output_dimension_3 = np.float32(output_dim_3)  
    model_data.con_layer_kernel_3 = np.float32(kernel_3)  
    model_data.con_layer_pooling_3 = np.float32(pooling_3)  

    model_data.full_connect_len = np.float32(full_connect_len)  
    model_data.classification_out = np.float32(classification_out)  

    model_data.weight_bias = []  

    
    for key in model_dict:
        model_data.weight_bias.append(np.float64((model_dict[key].cpu().numpy())))  

    return model_data



def lw_save_parameters(single, count, para_size, model_data, test_original_data):
    file_name = str(count) + '_weight_bias_image' + '_' \
                + str(int(model_data.image_size)) + '_' \
                + str(int(model_data.con_layer_input_dimension_1)) + '_' \
                + str(int(model_data.con_layer_output_dimension_1)) + '_' \
                + str(int(model_data.con_layer_kernel_1)) + '_' \
                + str(int(model_data.con_layer_pooling_1)) + '_' \
                + str(int(model_data.con_layer_input_dimension_2)) + '_' \
                + str(int(model_data.con_layer_output_dimension_2)) + '_' \
                + str(int(model_data.con_layer_kernel_2)) + '_' \
                + str(int(model_data.con_layer_pooling_2)) + '_' \
                + str(int(model_data.con_layer_input_dimension_3)) + '_' \
                + str(int(model_data.con_layer_output_dimension_3)) + '_' \
                + str(int(model_data.con_layer_kernel_3)) + '_' \
                + str(int(model_data.con_layer_pooling_3)) + '.bin'

    
    with open(file_name, 'wb') as file:
        
        file.write(model_data.head_len)
        file.write(model_data.con_layers)
        file.write(model_data.para_size)
        file.write(model_data.image_size)

        
        file.write(model_data.con_layer_input_dimension_1)
        file.write(model_data.con_layer_output_dimension_1)
        file.write(model_data.con_layer_kernel_1)
        file.write(model_data.con_layer_pooling_1)

        file.write(model_data.con_layer_input_dimension_2)
        file.write(model_data.con_layer_output_dimension_2)
        file.write(model_data.con_layer_kernel_2)
        file.write(model_data.con_layer_pooling_2)

        file.write(model_data.con_layer_input_dimension_3)
        file.write(model_data.con_layer_output_dimension_3)
        file.write(model_data.con_layer_kernel_3)
        file.write(model_data.con_layer_pooling_3)

        
        file.write(model_data.full_connect_len)
        file.write(model_data.classification_out)

        
        for i in range(len(model_data.weight_bias)):
            file.write(np.float32(model_data.weight_bias[i]))

        
        file.write(np.float32(test_original_data[0]))  


    shutil.move(os.path.join(os.getcwd(), file_name), mypath.model_path)  

    
    if single == 1:
        myfile.lw_delete_file(mypath.lw_visual_path, '.bin')  

        shutil.copy(os.path.join(mypath.model_path, file_name), mypath.lw_visual_path)  
        shutil.move(os.path.join(mypath.lw_visual_path, file_name),
                    os.path.join(mypath.lw_visual_path, '1_weight_bias_image.bin'))  

        shutil.copy(os.path.join(mypath.model_path, file_name), mypath.lw_visual_path)  

    return test_original_data[0], file_name



def lw_model_out_directly(test_image):
    load_model = torch.load('Best_CNN_model.pth')  
    test_image = torch.Tensor(test_image).unsqueeze(0).unsqueeze(0)  
    model_out = load_model(test_image).detach().numpy().flatten()

    return model_out



def lw_model_out_hierarchy(test_image, model_para):
    load_model = torch.load('Best_CNN_model.pth')  
    test_image = torch.Tensor(test_image).unsqueeze(0).unsqueeze(0)  

    
    conv2d_out_1 = functional.conv2d(test_image, torch.Tensor(model_para.weight_bias[0]),torch.Tensor(model_para.weight_bias[1]))
    relu_out_1 = functional.relu(conv2d_out_1)
    pool_out_1 = functional.max_pool2d(relu_out_1, int(model_para.con_layer_pooling_1))

    
    conv2d_out_2 = functional.conv2d(pool_out_1, torch.Tensor(model_para.weight_bias[2]),torch.Tensor(model_para.weight_bias[3]))
    relu_out_2 = functional.relu(conv2d_out_2)
    pool_out_2 = functional.max_pool2d(relu_out_2, int(model_para.con_layer_pooling_2))

    
    conv2d_out_3 = functional.conv2d(pool_out_2, torch.Tensor(model_para.weight_bias[4]),torch.Tensor(model_para.weight_bias[5]))
    relu_out_3 = functional.relu(conv2d_out_3)
    pool_out_3 = functional.max_pool2d(relu_out_3, int(model_para.con_layer_pooling_3))

    
    flatten_out = torch.flatten(pool_out_3)

    
    model_out = functional.linear(flatten_out, torch.Tensor(model_para.weight_bias[6]),torch.Tensor(model_para.weight_bias[7])).numpy()

    return model_out



def lw_model_out_layer(test_image, model_para):
    load_model = torch.load('Best_CNN_model.pth')  
    test_image = torch.Tensor(test_image).unsqueeze(0).unsqueeze(0)  

    model_out = []
    for i in range(len(load_model.myCNN_Convs)):
        model_out.append(load_model.myCNN_Convs[: i + 1](test_image).detach().numpy())  

    return model_out



def lw_model_out_detial(test_image, model_para):
    test_image = torch.Tensor(test_image).unsqueeze(0).unsqueeze(0)  

    
    conv2d_out_1 = functional.conv2d(test_image, torch.Tensor(model_para.weight_bias[0]),torch.Tensor(model_para.weight_bias[1]))
    relu_out_1 = functional.relu(conv2d_out_1)
    pool_out_1 = functional.max_pool2d(relu_out_1, int(model_para.con_layer_pooling_1))

    
    conv2d_out_2 = functional.conv2d(pool_out_1, torch.Tensor(model_para.weight_bias[2]),torch.Tensor(model_para.weight_bias[3]))
    relu_out_2 = functional.relu(conv2d_out_2)
    pool_out_2 = functional.max_pool2d(relu_out_2, int(model_para.con_layer_pooling_2))

    
    conv2d_out_3 = functional.conv2d(pool_out_2, torch.Tensor(model_para.weight_bias[4]),torch.Tensor(model_para.weight_bias[5]))
    relu_out_3 = functional.relu(conv2d_out_3)
    pool_out_3 = functional.max_pool2d(relu_out_3, int(model_para.con_layer_pooling_3))

    
    flatten_out = torch.flatten(pool_out_3)

    
    model_out = functional.linear(flatten_out, torch.Tensor(model_para.weight_bias[6]),torch.Tensor(model_para.weight_bias[7])).numpy()

    return model_out



def lw_computing_max_ram(model_para):
    ram_list = []
    for i in range(1, int(model_para.con_layers)):
        ram_size = lw_each_ram_occupy(model_para, i)  
        ram_list.append(ram_size)  
    max_ram = np.max(ram_list)

    ram_list.append(max_ram)  
    print("ÿ��RAM�Լ����RAM��")
    print(ram_list)

    return ram_list



def lw_each_ram_occupy(model_para, layer):
    conv_kernel_size = 0
    pool_kernel_size = 0
    input_dimension = 0
    output_dimension = 0
    bias_len = 0
    input_image_size = 0
    output_image_size = 0

    if layer == 1:
        conv_kernel_size = model_para.con_layer_kernel_1
        pool_kernel_size = model_para.con_layer_pooling_1
        input_dimension = model_para.con_layer_input_dimension_1
        output_dimension = model_para.con_layer_output_dimension_1
        bias_len = model_para.con_layer_output_dimension_1
        input_image_size = model_para.image_size
        output_image_size = lw_computing_output_imgs_size(model_para, 'pool', 1)

    if layer == 2:
        conv_kernel_size = model_para.con_layer_kernel_2
        pool_kernel_size = model_para.con_layer_pooling_2
        input_dimension = model_para.con_layer_input_dimension_2
        output_dimension = model_para.con_layer_output_dimension_2
        bias_len = model_para.con_layer_output_dimension_2
        input_image_size = lw_computing_output_imgs_size(model_para, 'pool', 1)
        output_image_size = lw_computing_output_imgs_size(model_para, 'conv', 2)

    if layer == 3:
        conv_kernel_size = model_para.con_layer_kernel_3
        pool_kernel_size = model_para.con_layer_pooling_3
        input_dimension = model_para.con_layer_input_dimension_3
        output_dimension = model_para.con_layer_output_dimension_3
        bias_len = model_para.con_layer_output_dimension_3
        input_image_size = lw_computing_output_imgs_size(model_para, 'conv', 2)
        output_image_size = lw_computing_output_imgs_size(model_para, 'pool', 3)

    head_size = model_para.head_len
    bias_size = bias_len

    weight_size = output_dimension * input_dimension * conv_kernel_size * conv_kernel_size

    total_input_images_size = input_dimension * input_image_size * input_image_size

    total_output_image_size = output_dimension * output_image_size * output_image_size

    one_filter_conv_kernel_size = input_dimension * conv_kernel_size * conv_kernel_size

    one_channel_conv_kernel_size = conv_kernel_size * conv_kernel_size

    one_channel_images_size = output_image_size * output_image_size

    ram_size_occupy = (head_size + bias_size + weight_size + total_input_images_size + total_output_image_size + \
                       one_filter_conv_kernel_size + one_channel_conv_kernel_size + one_channel_images_size) * 4

    return ram_size_occupy



def lw_computing_output_imgs_size(model_para, conv_pool, layer):
    output_image_size = model_para.image_size

    for i in range(1, layer + 1):
        if layer == 1:
            conv_kernel_size = model_para.con_layer_kernel_1
            pool_kernel_size = model_para.con_layer_pooling_1

        if layer == 2:
            conv_kernel_size = model_para.con_layer_kernel_2
            pool_kernel_size = model_para.con_layer_pooling_2

        if layer == 3:
            conv_kernel_size = model_para.con_layer_kernel_3
            pool_kernel_size = model_para.con_layer_pooling_3

        
        if i != layer:
            output_image_size = np.floor((output_image_size + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / float(1) + 1)
            output_image_size = np.floor(
                (output_image_size + 2 * 0 - 1 * (pool_kernel_size - 1) - 1) / float(pool_kernel_size) + 1)
        else:
            if conv_pool == 'conv':
                output_image_size = np.floor((output_image_size + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / float(1) + 1)
            else:  
                output_image_size = np.floor((output_image_size + 2 * 0 - 1 * (conv_kernel_size - 1) - 1) / float(1) + 1)
                output_image_size = np.floor((output_image_size + 2 * 0 - 1 * (pool_kernel_size - 1) - 1) / float(pool_kernel_size) + 1)

    return output_image_size
