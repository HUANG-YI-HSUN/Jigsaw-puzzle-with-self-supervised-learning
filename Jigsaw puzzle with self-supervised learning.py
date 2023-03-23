# This program is our final version
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import keras
import itertools
import random
import gc
from keras import backend as K 
from keras.layers import Input, Conv2D, Lambda, merge, Dense, Flatten,MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model, Sequential
from keras.regularizers import l1, l2
from keras import backend as K
from keras.optimizers import SGD,Adam
from keras.losses import binary_crossentropy
from keras.utils import plot_model
import keras.applications.resnet
from keras.applications.resnet50 import ResNet50 
from keras.applications.resnet_v2 import ResNet50V2
from keras.preprocessing import image_dataset_from_directory

# For GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the fourth GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


# For C(9! 100), but we only use forty categories
num_crops = 9
num_permutations = 100

P_hat = np.array(list(itertools.permutations(list(range(num_crops)), num_crops)))
new_array = np.zeros( (100,9) )
for i in range(num_permutations) :
    n = P_hat.shape[0]
    index = np.random.randint(n)
    np.delete(P_hat, index, axis=0)
    new_array[i] = P_hat[index]


train_test_split = 0.7

# We first stroe our data to file_names
file_names = [[]for i in range(9)]
filenames = ''

# We spilit a picture into 9 pieces and store in file_name1-9
for i in range (1 , 10) : 
    for parent, dirnames, filenames in os.walk("/data/data9_50000/data9/" + str(i)):  #三個引數：分別返回1.父目錄 2.所有資料夾名字（不含路徑） 3.所有檔名字
        file_names[i-1] = filenames
        file_names[i-1] = sorted(file_names[i-1])
    print(len(file_names[i-1]), 'check for every piece has same number of picture')

train_size = int(len(file_names[0]) * 0.7)


def get_batch (data_index, batch_size, random_data): # get a batch of train data  
    train_length = 40 * batch_size
    start=0
    end=train_size
    batch_x=[]
    batch_y = np.zeros((train_length, 40))

    batch_x.append(np.zeros((train_length, 64, 64, 3)))
    batch_x.append(np.zeros((train_length, 64, 64, 3)))
    batch_x.append(np.zeros((train_length, 64, 64, 3)))
    batch_x.append(np.zeros((train_length, 64, 64, 3)))
    batch_x.append(np.zeros((train_length, 64, 64, 3)))
    batch_x.append(np.zeros((train_length, 64, 64, 3)))
    batch_x.append(np.zeros((train_length, 64, 64, 3)))
    batch_x.append(np.zeros((train_length, 64, 64, 3)))
    batch_x.append(np.zeros((train_length, 64, 64, 3)))

    temp_array = np.zeros( (9,64,64,3) )
    i = 0 
    random_data_index = 0
    while( i != train_length ): #Read input
        for j in range ( 1 , 10 ) :
            a = keras.preprocessing.image.load_img("/data/data9_50000/data9/"+str(j)+"/"+file_names[j-1][random_data[data_index] * 10 + random_data_index], 
                                                    grayscale=False, color_mode="rgb", target_size=(64, 64), interpolation="nearest" )
    
            temp_array[j-1] = keras.preprocessing.image.img_to_array(a)  
            temp_array[j-1] = np.asarray(temp_array[j-1])/255.0

        random_data_index = random_data_index + 1
        
        #Forty categories and different combination
        batch_y[i][0] = 1
        batch_x[0][i] = temp_array[0]
        batch_x[1][i] = temp_array[1]
        batch_x[2][i] = temp_array[2]
        batch_x[3][i] = temp_array[3]
        batch_x[4][i] = temp_array[4]
        batch_x[5][i] = temp_array[5]
        batch_x[6][i] = temp_array[6] 
        batch_x[7][i] = temp_array[7]
        batch_x[8][i] = temp_array[8]
        
        i = i + 1
        batch_y[i][1] = 1
        batch_x[0][i] = temp_array[8]
        batch_x[1][i] = temp_array[7]
        batch_x[2][i] = temp_array[6]
        batch_x[3][i] = temp_array[5]
        batch_x[4][i] = temp_array[4]
        batch_x[5][i] = temp_array[3]
        batch_x[6][i] = temp_array[2] 
        batch_x[7][i] = temp_array[1]
        batch_x[8][i] = temp_array[0]
            
        i = i + 1
        batch_y[i][2] = 1
        batch_x[0][i] = temp_array[0]
        batch_x[1][i] = temp_array[1]
        batch_x[2][i] = temp_array[2]
        batch_x[3][i] = temp_array[6]
        batch_x[4][i] = temp_array[7]
        batch_x[5][i] = temp_array[8]
        batch_x[6][i] = temp_array[3] 
        batch_x[7][i] = temp_array[4]
        batch_x[8][i] = temp_array[5]
          
        i = i + 1
        batch_y[i][3] = 1
        batch_x[0][i] = temp_array[3]
        batch_x[1][i] = temp_array[4]
        batch_x[2][i] = temp_array[5]
        batch_x[3][i] = temp_array[0]
        batch_x[4][i] = temp_array[1]
        batch_x[5][i] = temp_array[2]
        batch_x[6][i] = temp_array[6] 
        batch_x[7][i] = temp_array[7]
        batch_x[8][i] = temp_array[8]
        
        i = i + 1
        batch_y[i][4] = 1
        batch_x[0][i] = temp_array[3]
        batch_x[1][i] = temp_array[4]
        batch_x[2][i] = temp_array[5]
        batch_x[3][i] = temp_array[6]
        batch_x[4][i] = temp_array[7]
        batch_x[5][i] = temp_array[8]
        batch_x[6][i] = temp_array[0] 
        batch_x[7][i] = temp_array[1]
        batch_x[8][i] = temp_array[2]
        
        i = i + 1
        batch_y[i][5] = 1
        batch_x[0][i] = temp_array[6]
        batch_x[1][i] = temp_array[7]
        batch_x[2][i] = temp_array[8]
        batch_x[3][i] = temp_array[0]
        batch_x[4][i] = temp_array[1]
        batch_x[5][i] = temp_array[2]
        batch_x[6][i] = temp_array[3] 
        batch_x[7][i] = temp_array[4]
        batch_x[8][i] = temp_array[5]
        
        i = i + 1
        batch_y[i][6] = 1
        batch_x[0][i] = temp_array[6]
        batch_x[1][i] = temp_array[7]
        batch_x[2][i] = temp_array[8]
        batch_x[3][i] = temp_array[3]
        batch_x[4][i] = temp_array[4]
        batch_x[5][i] = temp_array[5]
        batch_x[6][i] = temp_array[0] 
        batch_x[7][i] = temp_array[1]
        batch_x[8][i] = temp_array[2]
        
        i = i + 1
        batch_y[i][7] = 1
        batch_x[0][i] = temp_array[0]
        batch_x[1][i] = temp_array[2]
        batch_x[2][i] = temp_array[1]
        batch_x[3][i] = temp_array[3]
        batch_x[4][i] = temp_array[5]
        batch_x[5][i] = temp_array[4]
        batch_x[6][i] = temp_array[6] 
        batch_x[7][i] = temp_array[8]
        batch_x[8][i] = temp_array[7]
        
        i = i + 1
        batch_y[i][8] = 1
        batch_x[0][i] = temp_array[1]
        batch_x[1][i] = temp_array[0]
        batch_x[2][i] = temp_array[2]
        batch_x[3][i] = temp_array[4]
        batch_x[4][i] = temp_array[3]
        batch_x[5][i] = temp_array[5]
        batch_x[6][i] = temp_array[7] 
        batch_x[7][i] = temp_array[6]
        batch_x[8][i] = temp_array[8]
        
        i = i + 1
        batch_y[i][9] = 1
        batch_x[0][i] = temp_array[2]
        batch_x[1][i] = temp_array[1]
        batch_x[2][i] = temp_array[0]
        batch_x[3][i] = temp_array[5]
        batch_x[4][i] = temp_array[4]
        batch_x[5][i] = temp_array[3]
        batch_x[6][i] = temp_array[8] 
        batch_x[7][i] = temp_array[7]
        batch_x[8][i] = temp_array[6]
        
        i = i + 1 
        batch_y[i][10] = 1
        batch_x[0][i] = temp_array[2]
        batch_x[1][i] = temp_array[0]
        batch_x[2][i] = temp_array[1]
        batch_x[3][i] = temp_array[5]
        batch_x[4][i] = temp_array[3]
        batch_x[5][i] = temp_array[4]
        batch_x[6][i] = temp_array[8] 
        batch_x[7][i] = temp_array[6]
        batch_x[8][i] = temp_array[7]
        
        i = i + 1 
        batch_y[i][11] = 1
        batch_x[0][i] = temp_array[1]
        batch_x[1][i] = temp_array[2]
        batch_x[2][i] = temp_array[0]
        batch_x[3][i] = temp_array[4]
        batch_x[4][i] = temp_array[5]
        batch_x[5][i] = temp_array[3]
        batch_x[6][i] = temp_array[7] 
        batch_x[7][i] = temp_array[8]
        batch_x[8][i] = temp_array[6]
        
        i = i + 1
        batch_y[i][12] = 1
        batch_x[0][i] = temp_array[7]
        batch_x[1][i] = temp_array[0]
        batch_x[2][i] = temp_array[1]
        batch_x[3][i] = temp_array[5]
        batch_x[4][i] = temp_array[8]
        batch_x[5][i] = temp_array[4]
        batch_x[6][i] = temp_array[3] 
        batch_x[7][i] = temp_array[6]
        batch_x[8][i] = temp_array[2]
        
        i = i + 1 
        batch_y[i][13] = 1
        batch_x[0][i] = temp_array[5]
        batch_x[1][i] = temp_array[0]
        batch_x[2][i] = temp_array[1]
        batch_x[3][i] = temp_array[7]
        batch_x[4][i] = temp_array[2]
        batch_x[5][i] = temp_array[4]
        batch_x[6][i] = temp_array[3] 
        batch_x[7][i] = temp_array[6]
        batch_x[8][i] = temp_array[8]
        
        i = i + 1 
        batch_y[i][14] = 1
        batch_x[0][i] = temp_array[7]
        batch_x[1][i] = temp_array[0]
        batch_x[2][i] = temp_array[1]
        batch_x[3][i] = temp_array[5]
        batch_x[4][i] = temp_array[6]
        batch_x[5][i] = temp_array[4]
        batch_x[6][i] = temp_array[3] 
        batch_x[7][i] = temp_array[8]
        batch_x[8][i] = temp_array[2]
        
        i = i + 1
        batch_y[i][15] = 1
        batch_x[0][i] = temp_array[0]
        batch_x[1][i] = temp_array[2]
        batch_x[2][i] = temp_array[1]
        batch_x[3][i] = temp_array[7]
        batch_x[4][i] = temp_array[8]
        batch_x[5][i] = temp_array[4]
        batch_x[6][i] = temp_array[3] 
        batch_x[7][i] = temp_array[6]
        batch_x[8][i] = temp_array[5]
        
        i = i + 1
        batch_y[i][16] = 1
        batch_x[0][i] = temp_array[5]
        batch_x[1][i] = temp_array[0]
        batch_x[2][i] = temp_array[1]
        batch_x[3][i] = temp_array[4]
        batch_x[4][i] = temp_array[2]
        batch_x[5][i] = temp_array[7]
        batch_x[6][i] = temp_array[3] 
        batch_x[7][i] = temp_array[6]
        batch_x[8][i] = temp_array[8]
        
        i = i + 1
        batch_y[i][17] = 1
        batch_x[0][i] = temp_array[5]
        batch_x[1][i] = temp_array[0]
        batch_x[2][i] = temp_array[1]
        batch_x[3][i] = temp_array[3]
        batch_x[4][i] = temp_array[4]
        batch_x[5][i] = temp_array[8]
        batch_x[6][i] = temp_array[7] 
        batch_x[7][i] = temp_array[6]
        batch_x[8][i] = temp_array[2]
        
        i = i + 1
        batch_y[i][18] = 1
        batch_x[0][i] = temp_array[5]
        batch_x[1][i] = temp_array[0]
        batch_x[2][i] = temp_array[1]
        batch_x[3][i] = temp_array[8]
        batch_x[4][i] = temp_array[7]
        batch_x[5][i] = temp_array[3]
        batch_x[6][i] = temp_array[4] 
        batch_x[7][i] = temp_array[2]
        batch_x[8][i] = temp_array[6]
        
        i = i + 1
        batch_y[i][19] = 1
        batch_x[0][i] = temp_array[6]
        batch_x[1][i] = temp_array[0]
        batch_x[2][i] = temp_array[1]
        batch_x[3][i] = temp_array[7]
        batch_x[4][i] = temp_array[8]
        batch_x[5][i] = temp_array[4]
        batch_x[6][i] = temp_array[3] 
        batch_x[7][i] = temp_array[5]
        batch_x[8][i] = temp_array[2]
        
        i = i + 1
        batch_y[i][20] = 1
        batch_x[0][i] = temp_array[5]
        batch_x[1][i] = temp_array[6]
        batch_x[2][i] = temp_array[7]
        batch_x[3][i] = temp_array[8]
        batch_x[4][i] = temp_array[4]
        batch_x[5][i] = temp_array[3]
        batch_x[6][i] = temp_array[2] 
        batch_x[7][i] = temp_array[1]
        batch_x[8][i] = temp_array[0]
          
        i = i + 1
        batch_y[i][21] = 1
        batch_x[0][i] = temp_array[0]
        batch_x[1][i] = temp_array[1]
        batch_x[2][i] = temp_array[2]
        batch_x[3][i] = temp_array[3]
        batch_x[4][i] = temp_array[4]
        batch_x[5][i] = temp_array[8]
        batch_x[6][i] = temp_array[7] 
        batch_x[7][i] = temp_array[6]
        batch_x[8][i] = temp_array[5]
        
        i = i + 1
        batch_y[i][22] = 1
        batch_x[0][i] = temp_array[5]
        batch_x[1][i] = temp_array[1]
        batch_x[2][i] = temp_array[2]
        batch_x[3][i] = temp_array[3]
        batch_x[4][i] = temp_array[4]
        batch_x[5][i] = temp_array[8]
        batch_x[6][i] = temp_array[7] 
        batch_x[7][i] = temp_array[6]
        batch_x[8][i] = temp_array[0]
        
        i = i + 1
        batch_y[i][23] = 1
        batch_x[0][i] = temp_array[0]
        batch_x[1][i] = temp_array[1]
        batch_x[2][i] = temp_array[2]
        batch_x[3][i] = temp_array[3]
        batch_x[4][i] = temp_array[8]
        batch_x[5][i] = temp_array[4]
        batch_x[6][i] = temp_array[7] 
        batch_x[7][i] = temp_array[6]
        batch_x[8][i] = temp_array[5]
        
        i = i + 1
        batch_y[i][24] = 1
        batch_x[0][i] = temp_array[1]
        batch_x[1][i] = temp_array[0]
        batch_x[2][i] = temp_array[3]
        batch_x[3][i] = temp_array[2]
        batch_x[4][i] = temp_array[8]
        batch_x[5][i] = temp_array[4]
        batch_x[6][i] = temp_array[7] 
        batch_x[7][i] = temp_array[6]
        batch_x[8][i] = temp_array[5]
        
        i = i + 1
        batch_y[i][25] = 1
        batch_x[0][i] = temp_array[0]
        batch_x[1][i] = temp_array[1]
        batch_x[2][i] = temp_array[2]
        batch_x[3][i] = temp_array[3]
        batch_x[4][i] = temp_array[8]
        batch_x[5][i] = temp_array[4]
        batch_x[6][i] = temp_array[7] 
        batch_x[7][i] = temp_array[5]
        batch_x[8][i] = temp_array[6]
        
        i = i + 1
        batch_y[i][26] = 1
        batch_x[0][i] = temp_array[3]
        batch_x[1][i] = temp_array[1]
        batch_x[2][i] = temp_array[2]
        batch_x[3][i] = temp_array[0]
        batch_x[4][i] = temp_array[8]
        batch_x[5][i] = temp_array[4]
        batch_x[6][i] = temp_array[7] 
        batch_x[7][i] = temp_array[6]
        batch_x[8][i] = temp_array[5]
        
        i = i + 1
        batch_y[i][27] = 1
        batch_x[0][i] = temp_array[0]
        batch_x[1][i] = temp_array[1]
        batch_x[2][i] = temp_array[2]
        batch_x[3][i] = temp_array[7]
        batch_x[4][i] = temp_array[8]
        batch_x[5][i] = temp_array[4]
        batch_x[6][i] = temp_array[3] 
        batch_x[7][i] = temp_array[6]
        batch_x[8][i] = temp_array[5]
        
        i = i + 1
        batch_y[i][28] = 1
        batch_x[0][i] = temp_array[8]
        batch_x[1][i] = temp_array[7]
        batch_x[2][i] = temp_array[6]
        batch_x[3][i] = temp_array[2]
        batch_x[4][i] = temp_array[1]
        batch_x[5][i] = temp_array[0]
        batch_x[6][i] = temp_array[5] 
        batch_x[7][i] = temp_array[4]
        batch_x[8][i] = temp_array[3]

        i = i + 1
        batch_y[i][29] = 1
        batch_x[0][i] = temp_array[5]
        batch_x[1][i] = temp_array[4]
        batch_x[2][i] = temp_array[3]
        batch_x[3][i] = temp_array[8]
        batch_x[4][i] = temp_array[7]
        batch_x[5][i] = temp_array[6]
        batch_x[6][i] = temp_array[2] 
        batch_x[7][i] = temp_array[1]
        batch_x[8][i] = temp_array[0]

        i = i + 1
        batch_y[i][30] = 1
        batch_x[0][i] = temp_array[5]
        batch_x[1][i] = temp_array[4]
        batch_x[2][i] = temp_array[3]
        batch_x[3][i] = temp_array[2]
        batch_x[4][i] = temp_array[1]
        batch_x[5][i] = temp_array[0]
        batch_x[6][i] = temp_array[8] 
        batch_x[7][i] = temp_array[7]
        batch_x[8][i] = temp_array[6]

        i = i + 1
        batch_y[i][31] = 1
        batch_x[0][i] = temp_array[2]
        batch_x[1][i] = temp_array[1]
        batch_x[2][i] = temp_array[0]
        batch_x[3][i] = temp_array[8]
        batch_x[4][i] = temp_array[7]
        batch_x[5][i] = temp_array[6]
        batch_x[6][i] = temp_array[5] 
        batch_x[7][i] = temp_array[4]
        batch_x[8][i] = temp_array[3]

        i = i + 1
        batch_y[i][32] = 1
        batch_x[0][i] = temp_array[2]
        batch_x[1][i] = temp_array[1]
        batch_x[2][i] = temp_array[0]
        batch_x[3][i] = temp_array[5]
        batch_x[4][i] = temp_array[4]
        batch_x[5][i] = temp_array[3]
        batch_x[6][i] = temp_array[8] 
        batch_x[7][i] = temp_array[7]
        batch_x[8][i] = temp_array[6]

        i = i + 1
        batch_y[i][33] = 1
        batch_x[0][i] = temp_array[7]
        batch_x[1][i] = temp_array[6]
        batch_x[2][i] = temp_array[8]
        batch_x[3][i] = temp_array[3]
        batch_x[4][i] = temp_array[4]
        batch_x[5][i] = temp_array[5]
        batch_x[6][i] = temp_array[0] 
        batch_x[7][i] = temp_array[1]
        batch_x[8][i] = temp_array[2]

        i = i + 1
        batch_y[i][34] = 1
        batch_x[0][i] = temp_array[8]
        batch_x[1][i] = temp_array[6]
        batch_x[2][i] = temp_array[7]
        batch_x[3][i] = temp_array[5]
        batch_x[4][i] = temp_array[3]
        batch_x[5][i] = temp_array[4]
        batch_x[6][i] = temp_array[2] 
        batch_x[7][i] = temp_array[0]
        batch_x[8][i] = temp_array[1]

        i = i + 1
        batch_y[i][35] = 1
        batch_x[0][i] = temp_array[7]
        batch_x[1][i] = temp_array[8]
        batch_x[2][i] = temp_array[6]
        batch_x[3][i] = temp_array[4]
        batch_x[4][i] = temp_array[5]
        batch_x[5][i] = temp_array[3]
        batch_x[6][i] = temp_array[1] 
        batch_x[7][i] = temp_array[2]
        batch_x[8][i] = temp_array[0]

        i = i + 1
        batch_y[i][36] = 1
        batch_x[0][i] = temp_array[7]
        batch_x[1][i] = temp_array[6]
        batch_x[2][i] = temp_array[8]
        batch_x[3][i] = temp_array[4]
        batch_x[4][i] = temp_array[3]
        batch_x[5][i] = temp_array[5]
        batch_x[6][i] = temp_array[1] 
        batch_x[7][i] = temp_array[0]
        batch_x[8][i] = temp_array[2]

        i = i + 1
        batch_y[i][37] = 1
        batch_x[0][i] = temp_array[6]
        batch_x[1][i] = temp_array[8]
        batch_x[2][i] = temp_array[7]
        batch_x[3][i] = temp_array[3]
        batch_x[4][i] = temp_array[5]
        batch_x[5][i] = temp_array[4]
        batch_x[6][i] = temp_array[0] 
        batch_x[7][i] = temp_array[2]
        batch_x[8][i] = temp_array[1]

        i = i + 1
        batch_y[i][38] = 1
        batch_x[0][i] = temp_array[8]
        batch_x[1][i] = temp_array[1]
        batch_x[2][i] = temp_array[6]
        batch_x[3][i] = temp_array[3]
        batch_x[4][i] = temp_array[4]
        batch_x[5][i] = temp_array[5]
        batch_x[6][i] = temp_array[2] 
        batch_x[7][i] = temp_array[7]
        batch_x[8][i] = temp_array[0]

        i = i + 1
        batch_y[i][39] = 1
        batch_x[0][i] = temp_array[0]
        batch_x[1][i] = temp_array[7]
        batch_x[2][i] = temp_array[2]
        batch_x[3][i] = temp_array[5]
        batch_x[4][i] = temp_array[4]
        batch_x[5][i] = temp_array[3]
        batch_x[6][i] = temp_array[6] 
        batch_x[7][i] = temp_array[1]
        batch_x[8][i] = temp_array[8]
        
        i = i + 1
    del temp_array # For memory 
    return(batch_x, batch_y)

#Building a  model
input_shape=(64, 64, 3)
input1 = Input(input_shape)
input2 = Input(input_shape)
input3 = Input(input_shape)
input4 = Input(input_shape)
input5 = Input(input_shape)
input6 = Input(input_shape)
input7 = Input(input_shape)
input8 = Input(input_shape)
input9 = Input(input_shape)

W_init = keras.initializers.RandomNormal(mean = 0.0, stddev = 1e-2)
b_init = keras.initializers.RandomNormal(mean = 0.5, stddev = 1e-2)

model = ResNet50V2(weights='imagenet', include_top=False, input_shape=input_shape)

for layer in model.layers:
    layer.trainable = False
    
model.summary()

encoded_1 = model(input1)
encoded_2 = model(input2)
encoded_3 = model(input3)
encoded_4 = model(input4)
encoded_5 = model(input5)
encoded_6 = model(input6)
encoded_7 = model(input7)
encoded_8 = model(input8)
encoded_9 = model(input9)

last_1 = keras.layers.GlobalAveragePooling2D()(encoded_1)
last_2 = keras.layers.GlobalAveragePooling2D()(encoded_2)
last_3 = keras.layers.GlobalAveragePooling2D()(encoded_3)
last_4 = keras.layers.GlobalAveragePooling2D()(encoded_4)
last_5 = keras.layers.GlobalAveragePooling2D()(encoded_5)
last_6 = keras.layers.GlobalAveragePooling2D()(encoded_6)
last_7 = keras.layers.GlobalAveragePooling2D()(encoded_7)
last_8 = keras.layers.GlobalAveragePooling2D()(encoded_8)
last_9 = keras.layers.GlobalAveragePooling2D()(encoded_9)

f = keras.layers.Concatenate()([last_1, last_2, last_3, last_4, last_5, last_6, last_7, last_8, last_9])

fc7 = Dense(4608, activation='sigmoid', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001), kernel_initializer=W_init, bias_initializer=b_init)(f)
fc8 = Dense(4096, activation='sigmoid', kernel_regularizer=l2(0.0001), bias_regularizer=l2(0.0001), kernel_initializer=W_init, bias_initializer=b_init)(fc7)
prediction = Dense(40, activation='sigmoid')(fc8)
siamese_net = Model([input1, input2, input3, input4, input5, input6, input7, input8, input9], prediction)

optimizer= Adam(0.0006)
siamese_net.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=[keras.metrics.CategoricalAccuracy()])

#siamese_net.load_weights('my_model_weights.h5')
siamese_net.summary()
#plot_model(siamese_net, show_shapes=True, show_layer_names=True) # is for model's strcture

def get_val(val_data_index, batch_size, random_data, val_ii): #Get validation data
    val_length = batch_size
    val_x = []
    val_y = np.zeros((val_length, 40))

    val_x.append(np.zeros((val_length, 64, 64, 3)))
    val_x.append(np.zeros((val_length, 64, 64, 3)))
    val_x.append(np.zeros((val_length, 64, 64, 3)))
    val_x.append(np.zeros((val_length, 64, 64, 3)))
    val_x.append(np.zeros((val_length, 64, 64, 3)))
    val_x.append(np.zeros((val_length, 64, 64, 3)))
    val_x.append(np.zeros((val_length, 64, 64, 3)))
    val_x.append(np.zeros((val_length, 64, 64, 3)))
    val_x.append(np.zeros((val_length, 64, 64, 3)))

    temp_array = np.zeros( (9,64,64,3) )
    for i in range(0, val_length):
        for j in range ( 1 , 10 ) :
            a = keras.preprocessing.image.load_img("/data/data9_50000/data9/"+str(j)+"/"+file_names[j-1][random_data[val_data_index] * 10  + val_ii], 
                                                    grayscale=False, color_mode="rgb", target_size=(64, 64), interpolation="nearest" )
    
            temp_array[j-1] = keras.preprocessing.image.img_to_array(a)
            temp_array[j-1] = np.asarray(temp_array[j-1])/255.0
             
        
        val_data_index = val_data_index + 1
        random = np.random.randint(40)
                
        #Get 1 category for validation
        if(random == 0) :
            val_y[i][0] = 1
            val_x[0][i] = temp_array[0]
            val_x[1][i] = temp_array[1]
            val_x[2][i] = temp_array[2]
            val_x[3][i] = temp_array[3]
            val_x[4][i] = temp_array[4]
            val_x[5][i] = temp_array[5]
            val_x[6][i] = temp_array[6] 
            val_x[7][i] = temp_array[7]
            val_x[8][i] = temp_array[8]
        
        elif(random == 1) :
            val_y[i][1] = 1
            val_x[0][i] = temp_array[8]
            val_x[1][i] = temp_array[7]
            val_x[2][i] = temp_array[6]
            val_x[3][i] = temp_array[5]
            val_x[4][i] = temp_array[4]
            val_x[5][i] = temp_array[3]
            val_x[6][i] = temp_array[2] 
            val_x[7][i] = temp_array[1]
            val_x[8][i] = temp_array[0]
        
        elif(random == 2) :
            val_y[i][2] = 1
            val_x[0][i] = temp_array[0]
            val_x[1][i] = temp_array[1]
            val_x[2][i] = temp_array[2]
            val_x[3][i] = temp_array[6]
            val_x[4][i] = temp_array[7]
            val_x[5][i] = temp_array[8]
            val_x[6][i] = temp_array[3] 
            val_x[7][i] = temp_array[4]
            val_x[8][i] = temp_array[5]

        elif (random == 3) :
            val_y[i][3] = 1
            val_x[0][i] = temp_array[3]
            val_x[1][i] = temp_array[4]
            val_x[2][i] = temp_array[5]
            val_x[3][i] = temp_array[0]
            val_x[4][i] = temp_array[1]
            val_x[5][i] = temp_array[2]
            val_x[6][i] = temp_array[6] 
            val_x[7][i] = temp_array[7]
            val_x[8][i] = temp_array[8]
            
        elif (random == 4) :
            val_y[i][4] = 1
            val_x[0][i] = temp_array[3]
            val_x[1][i] = temp_array[4]
            val_x[2][i] = temp_array[5]
            val_x[3][i] = temp_array[6]
            val_x[4][i] = temp_array[7]
            val_x[5][i] = temp_array[8]
            val_x[6][i] = temp_array[0] 
            val_x[7][i] = temp_array[1]
            val_x[8][i] = temp_array[2]
            
        elif (random == 5) :
            val_y[i][5] = 1
            val_x[0][i] = temp_array[6]
            val_x[1][i] = temp_array[7]
            val_x[2][i] = temp_array[8]
            val_x[3][i] = temp_array[0]
            val_x[4][i] = temp_array[1]
            val_x[5][i] = temp_array[2]
            val_x[6][i] = temp_array[3] 
            val_x[7][i] = temp_array[4]
            val_x[8][i] = temp_array[5]
            
        elif (random == 6) :
            val_y[i][6] = 1
            val_x[0][i] = temp_array[6]
            val_x[1][i] = temp_array[7]
            val_x[2][i] = temp_array[8]
            val_x[3][i] = temp_array[3]
            val_x[4][i] = temp_array[4]
            val_x[5][i] = temp_array[5]
            val_x[6][i] = temp_array[0] 
            val_x[7][i] = temp_array[1]
            val_x[8][i] = temp_array[2]
            
        elif (random == 7) :
            val_y[i][7] = 1
            val_x[0][i] = temp_array[0]
            val_x[1][i] = temp_array[2]
            val_x[2][i] = temp_array[1]
            val_x[3][i] = temp_array[3]
            val_x[4][i] = temp_array[5]
            val_x[5][i] = temp_array[4]
            val_x[6][i] = temp_array[6] 
            val_x[7][i] = temp_array[8]
            val_x[8][i] = temp_array[7]
            
        elif (random == 8) :
            val_y[i][8] = 1
            val_x[0][i] = temp_array[1]
            val_x[1][i] = temp_array[0]
            val_x[2][i] = temp_array[2]
            val_x[3][i] = temp_array[4]
            val_x[4][i] = temp_array[3]
            val_x[5][i] = temp_array[5]
            val_x[6][i] = temp_array[7] 
            val_x[7][i] = temp_array[6]
            val_x[8][i] = temp_array[8]
            
        elif (random == 9) :
            val_y[i][9] = 1
            val_x[0][i] = temp_array[2]
            val_x[1][i] = temp_array[1]
            val_x[2][i] = temp_array[0]
            val_x[3][i] = temp_array[5]
            val_x[4][i] = temp_array[4]
            val_x[5][i] = temp_array[3]
            val_x[6][i] = temp_array[8] 
            val_x[7][i] = temp_array[7]
            val_x[8][i] = temp_array[6]
            
        elif (random == 10) :
            val_y[i][10] = 1
            val_x[0][i] = temp_array[2]
            val_x[1][i] = temp_array[0]
            val_x[2][i] = temp_array[1]
            val_x[3][i] = temp_array[5]
            val_x[4][i] = temp_array[3]
            val_x[5][i] = temp_array[4]
            val_x[6][i] = temp_array[8] 
            val_x[7][i] = temp_array[6]
            val_x[8][i] = temp_array[7]
            
        elif (random == 11) :
            val_y[i][11] = 1
            val_x[0][i] = temp_array[1]
            val_x[1][i] = temp_array[2]
            val_x[2][i] = temp_array[0]
            val_x[3][i] = temp_array[4]
            val_x[4][i] = temp_array[5]
            val_x[5][i] = temp_array[3]
            val_x[6][i] = temp_array[7] 
            val_x[7][i] = temp_array[8]
            val_x[8][i] = temp_array[6]
        
        elif (random == 12) :
            val_y[i][12] = 1
            val_x[0][i] = temp_array[7]
            val_x[1][i] = temp_array[0]
            val_x[2][i] = temp_array[1]
            val_x[3][i] = temp_array[5]
            val_x[4][i] = temp_array[8]
            val_x[5][i] = temp_array[4]
            val_x[6][i] = temp_array[3] 
            val_x[7][i] = temp_array[6]
            val_x[8][i] = temp_array[2]
            
        elif (random == 13) :
            val_y[i][13] = 1
            val_x[0][i] = temp_array[5]
            val_x[1][i] = temp_array[0]
            val_x[2][i] = temp_array[1]
            val_x[3][i] = temp_array[7]
            val_x[4][i] = temp_array[2]
            val_x[5][i] = temp_array[4]
            val_x[6][i] = temp_array[3] 
            val_x[7][i] = temp_array[6]
            val_x[8][i] = temp_array[8]
            
        elif (random == 14) :
            val_y[i][14] = 1
            val_x[0][i] = temp_array[7]
            val_x[1][i] = temp_array[0]
            val_x[2][i] = temp_array[1]
            val_x[3][i] = temp_array[5]
            val_x[4][i] = temp_array[6]
            val_x[5][i] = temp_array[4]
            val_x[6][i] = temp_array[3] 
            val_x[7][i] = temp_array[8]
            val_x[8][i] = temp_array[2]
            
        elif (random == 15) :
            val_y[i][15] = 1
            val_x[0][i] = temp_array[0]
            val_x[1][i] = temp_array[2]
            val_x[2][i] = temp_array[1]
            val_x[3][i] = temp_array[7]
            val_x[4][i] = temp_array[8]
            val_x[5][i] = temp_array[4]
            val_x[6][i] = temp_array[3] 
            val_x[7][i] = temp_array[6]
            val_x[8][i] = temp_array[5]
            
        elif (random == 16) :
            val_y[i][16] = 1
            val_x[0][i] = temp_array[5]
            val_x[1][i] = temp_array[0]
            val_x[2][i] = temp_array[1]
            val_x[3][i] = temp_array[4]
            val_x[4][i] = temp_array[2]
            val_x[5][i] = temp_array[7]
            val_x[6][i] = temp_array[3] 
            val_x[7][i] = temp_array[6]
            val_x[8][i] = temp_array[8]
            
        elif (random == 17) :
            val_y[i][17] = 1
            val_x[0][i] = temp_array[5]
            val_x[1][i] = temp_array[0]
            val_x[2][i] = temp_array[1]
            val_x[3][i] = temp_array[3]
            val_x[4][i] = temp_array[4]
            val_x[5][i] = temp_array[8]
            val_x[6][i] = temp_array[7] 
            val_x[7][i] = temp_array[6]
            val_x[8][i] = temp_array[2]
            
        elif (random == 18) :
            val_y[i][18] = 1
            val_x[0][i] = temp_array[5]
            val_x[1][i] = temp_array[0]
            val_x[2][i] = temp_array[1]
            val_x[3][i] = temp_array[8]
            val_x[4][i] = temp_array[7]
            val_x[5][i] = temp_array[3]
            val_x[6][i] = temp_array[4] 
            val_x[7][i] = temp_array[2]
            val_x[8][i] = temp_array[6]
            
        elif (random == 19) :
            val_y[i][19] = 1
            val_x[0][i] = temp_array[6]
            val_x[1][i] = temp_array[0]
            val_x[2][i] = temp_array[1]
            val_x[3][i] = temp_array[7]
            val_x[4][i] = temp_array[8]
            val_x[5][i] = temp_array[4]
            val_x[6][i] = temp_array[3] 
            val_x[7][i] = temp_array[5]
            val_x[8][i] = temp_array[2]

        elif (random == 20) :
            val_y[i][20] = 1
            val_x[0][i] = temp_array[5]
            val_x[1][i] = temp_array[6]
            val_x[2][i] = temp_array[7]
            val_x[3][i] = temp_array[8]
            val_x[4][i] = temp_array[4]
            val_x[5][i] = temp_array[3]
            val_x[6][i] = temp_array[2] 
            val_x[7][i] = temp_array[1]
            val_x[8][i] = temp_array[0]
            
        elif (random == 21) :
            val_y[i][21] = 1
            val_x[0][i] = temp_array[0]
            val_x[1][i] = temp_array[1]
            val_x[2][i] = temp_array[2]
            val_x[3][i] = temp_array[3]
            val_x[4][i] = temp_array[4]
            val_x[5][i] = temp_array[8]
            val_x[6][i] = temp_array[7] 
            val_x[7][i] = temp_array[6]
            val_x[8][i] = temp_array[5]
        
        elif (random == 22) :
            val_y[i][22] = 1
            val_x[0][i] = temp_array[5]
            val_x[1][i] = temp_array[1]
            val_x[2][i] = temp_array[2]
            val_x[3][i] = temp_array[3]
            val_x[4][i] = temp_array[4]
            val_x[5][i] = temp_array[8]
            val_x[6][i] = temp_array[7] 
            val_x[7][i] = temp_array[6]
            val_x[8][i] = temp_array[0]
        
        elif (random == 23) :
            val_y[i][23] = 1
            val_x[0][i] = temp_array[0]
            val_x[1][i] = temp_array[1]
            val_x[2][i] = temp_array[2]
            val_x[3][i] = temp_array[3]
            val_x[4][i] = temp_array[8]
            val_x[5][i] = temp_array[4]
            val_x[6][i] = temp_array[7] 
            val_x[7][i] = temp_array[6]
            val_x[8][i] = temp_array[5]
        
        elif (random == 24) :
            val_y[i][24] = 1
            val_x[0][i] = temp_array[1]
            val_x[1][i] = temp_array[0]
            val_x[2][i] = temp_array[3]
            val_x[3][i] = temp_array[2]
            val_x[4][i] = temp_array[8]
            val_x[5][i] = temp_array[4]
            val_x[6][i] = temp_array[7] 
            val_x[7][i] = temp_array[6]
            val_x[8][i] = temp_array[5]
        
        elif (random == 25) :
            val_y[i][25] = 1
            val_x[0][i] = temp_array[0]
            val_x[1][i] = temp_array[1]
            val_x[2][i] = temp_array[2]
            val_x[3][i] = temp_array[3]
            val_x[4][i] = temp_array[8]
            val_x[5][i] = temp_array[4]
            val_x[6][i] = temp_array[7] 
            val_x[7][i] = temp_array[5]
            val_x[8][i] = temp_array[6]
        
        elif (random == 26) :
            val_y[i][26] = 1
            val_x[0][i] = temp_array[3]
            val_x[1][i] = temp_array[1]
            val_x[2][i] = temp_array[2]
            val_x[3][i] = temp_array[0]
            val_x[4][i] = temp_array[8]
            val_x[5][i] = temp_array[4]
            val_x[6][i] = temp_array[7] 
            val_x[7][i] = temp_array[6]
            val_x[8][i] = temp_array[5]
        
        elif (random == 27) :
            val_y[i][27] = 1
            val_x[0][i] = temp_array[0]
            val_x[1][i] = temp_array[1]
            val_x[2][i] = temp_array[2]
            val_x[3][i] = temp_array[7]
            val_x[4][i] = temp_array[8]
            val_x[5][i] = temp_array[4]
            val_x[6][i] = temp_array[3] 
            val_x[7][i] = temp_array[6]
            val_x[8][i] = temp_array[5]
        
        elif (random == 28) :
            val_y[i][28] = 1
            val_x[0][i] = temp_array[8]
            val_x[1][i] = temp_array[7]
            val_x[2][i] = temp_array[6]
            val_x[3][i] = temp_array[2]
            val_x[4][i] = temp_array[1]
            val_x[5][i] = temp_array[0]
            val_x[6][i] = temp_array[5] 
            val_x[7][i] = temp_array[4]
            val_x[8][i] = temp_array[3]

        elif (random == 29) :
            val_y[i][29] = 1
            val_x[0][i] = temp_array[5]
            val_x[1][i] = temp_array[4]
            val_x[2][i] = temp_array[3]
            val_x[3][i] = temp_array[8]
            val_x[4][i] = temp_array[7]
            val_x[5][i] = temp_array[6]
            val_x[6][i] = temp_array[2] 
            val_x[7][i] = temp_array[1]
            val_x[8][i] = temp_array[0]

        elif (random == 30) :
            val_y[i][30] = 1
            val_x[0][i] = temp_array[5]
            val_x[1][i] = temp_array[4]
            val_x[2][i] = temp_array[3]
            val_x[3][i] = temp_array[2]
            val_x[4][i] = temp_array[1]
            val_x[5][i] = temp_array[0]
            val_x[6][i] = temp_array[8] 
            val_x[7][i] = temp_array[7]
            val_x[8][i] = temp_array[6]

        elif (random == 31) :
            val_y[i][31] = 1
            val_x[0][i] = temp_array[2]
            val_x[1][i] = temp_array[1]
            val_x[2][i] = temp_array[0]
            val_x[3][i] = temp_array[8]
            val_x[4][i] = temp_array[7]
            val_x[5][i] = temp_array[6]
            val_x[6][i] = temp_array[5] 
            val_x[7][i] = temp_array[4]
            val_x[8][i] = temp_array[3]

        elif (random == 32) :
            val_y[i][32] = 1
            val_x[0][i] = temp_array[2]
            val_x[1][i] = temp_array[1]
            val_x[2][i] = temp_array[0]
            val_x[3][i] = temp_array[5]
            val_x[4][i] = temp_array[4]
            val_x[5][i] = temp_array[3]
            val_x[6][i] = temp_array[8] 
            val_x[7][i] = temp_array[7]
            val_x[8][i] = temp_array[6]

        elif (random == 33) :
            val_y[i][33] = 1
            val_x[0][i] = temp_array[7]
            val_x[1][i] = temp_array[6]
            val_x[2][i] = temp_array[8]
            val_x[3][i] = temp_array[3]
            val_x[4][i] = temp_array[4]
            val_x[5][i] = temp_array[5]
            val_x[6][i] = temp_array[0] 
            val_x[7][i] = temp_array[1]
            val_x[8][i] = temp_array[2]

        elif (random == 34) :
            val_y[i][34] = 1
            val_x[0][i] = temp_array[8]
            val_x[1][i] = temp_array[6]
            val_x[2][i] = temp_array[7]
            val_x[3][i] = temp_array[5]
            val_x[4][i] = temp_array[3]
            val_x[5][i] = temp_array[4]
            val_x[6][i] = temp_array[2] 
            val_x[7][i] = temp_array[0]
            val_x[8][i] = temp_array[1]

        elif (random == 35) :
            val_y[i][35] = 1
            val_x[0][i] = temp_array[7]
            val_x[1][i] = temp_array[8]
            val_x[2][i] = temp_array[6]
            val_x[3][i] = temp_array[4]
            val_x[4][i] = temp_array[5]
            val_x[5][i] = temp_array[3]
            val_x[6][i] = temp_array[1] 
            val_x[7][i] = temp_array[2]
            val_x[8][i] = temp_array[0]

        elif (random == 36) :
            val_y[i][36] = 1
            val_x[0][i] = temp_array[7]
            val_x[1][i] = temp_array[6]
            val_x[2][i] = temp_array[8]
            val_x[3][i] = temp_array[4]
            val_x[4][i] = temp_array[3]
            val_x[5][i] = temp_array[5]
            val_x[6][i] = temp_array[1] 
            val_x[7][i] = temp_array[0]
            val_x[8][i] = temp_array[2]

        elif (random == 37) :
            val_y[i][37] = 1
            val_x[0][i] = temp_array[6]
            val_x[1][i] = temp_array[8]
            val_x[2][i] = temp_array[7]
            val_x[3][i] = temp_array[3]
            val_x[4][i] = temp_array[5]
            val_x[5][i] = temp_array[4]
            val_x[6][i] = temp_array[0] 
            val_x[7][i] = temp_array[2]
            val_x[8][i] = temp_array[1]

        elif (random == 38) :
            val_y[i][38] = 1
            val_x[0][i] = temp_array[8]
            val_x[1][i] = temp_array[1]
            val_x[2][i] = temp_array[6]
            val_x[3][i] = temp_array[3]
            val_x[4][i] = temp_array[4]
            val_x[5][i] = temp_array[5]
            val_x[6][i] = temp_array[2] 
            val_x[7][i] = temp_array[7]
            val_x[8][i] = temp_array[0]

        elif (random == 39) :
            val_y[i][39] = 1
            val_x[0][i] = temp_array[0]
            val_x[1][i] = temp_array[7]
            val_x[2][i] = temp_array[2]
            val_x[3][i] = temp_array[5]
            val_x[4][i] = temp_array[4]
            val_x[5][i] = temp_array[3]
            val_x[6][i] = temp_array[6] 
            val_x[7][i] = temp_array[1]
            val_x[8][i] = temp_array[8]

    del temp_array#For memory
    return(val_x, val_y)


epochs = 31
n_way = 20
n_val = 100
valsize = 500
batch_size = 10

loss_his = []
accuracy_his = []
train_loss_his = []
train_acc_his = []

each_round = int(len(file_names[0]) * 0.7 / batch_size)
val_each_round = int(len(file_names[0]) * 0.3 / batch_size)
val_loss_total = 0 
acc_loss_total = 0
train_loss_total = 0
train_acc_total = 0
data_index = 0
val_ii = 0
val_data_index = int(len(file_names[0]) * 0.7)


for epoch in range(1,epochs):
    data_index = 0
    val_data_index = int(len(file_names[0])/batch_size * 0.7)
    random_data = random.sample( range(0, int(len(file_names[0])/batch_size)), int(len(file_names[0])/batch_size))
    for each_epoch in range (0, each_round ) : 
        batch_x, batch_y = get_batch(data_index, batch_size, random_data)
        loss, acc = siamese_net.train_on_batch(batch_x, batch_y) 
        train_loss_total = train_loss_total + loss
        train_acc_total = train_acc_total + acc
        data_index = data_index + 1
        print('Batch:', each_epoch, ',  Loss:',loss, ',      Accuracy:', acc)
        del batch_x
        del batch_y
        if ( each_epoch % 100 == 0 ) : #For memory
            gc.collect()
    train_loss_total = train_loss_total / each_round
    train_acc_total = train_acc_total / each_round
    train_loss_his.append( str(train_loss_total) )
    loss_his.append( " " )
    train_acc_his.append( str(train_acc_total) )
    accuracy_his.append( " " )  
  
    print('-----------------------------')
    for k in range(0, val_each_round) : 
        val_x, val_y = get_val(val_data_index,batch_size, random_data, val_ii)
        val_ii = val_ii + 1
        if ( val_ii == 10 ) :
          val_ii = 0
          val_data_index = val_data_index + 1
        val_loss, val_acc = siamese_net.test_on_batch( val_x, val_y )
        val_loss_total = val_loss_total + val_loss
        acc_loss_total = acc_loss_total + val_acc
        del val_x
        del val_y
        if ( each_epoch % 100 == 0 ) :#For memory
            gc.collect()
    
    val_loss = val_loss_total / val_each_round
    val_acc = acc_loss_total / val_each_round
    loss_his.append( str(val_loss) )
    loss_his.append( " " )
    accuracy_his.append( str(val_acc) )
    accuracy_his.append( " " )
    val_loss_total = 0 
    acc_loss_total = 0 
    train_loss_total = 0
    train_acc_total = 0
    print('Epoch:',epoch, ',    Validation Loss:',val_loss, ',     Validation Accuracy:', val_acc)
    print('-----------------------------')
    gc.collect()#For memory


#This is for output
print( loss_his, 'Valadation Loss' )     
print( accuracy_his, 'Valadation Acc' )
f1 = open("file1.txt", "w")
f1.write("Valadation Loss ")
f1.writelines(loss_his)
f1.write("\n")
f1.write("Valadation Acc ")
f1.writelines(accuracy_his)
f1.write("\n")
f1.write("Train Acc ")
f1.writelines(train_acc_his)
f1.write("\n")
f1.write("Train Loss ")
f1.writelines(train_loss_his)
f1.close()
siamese_net.save_weights('my_model_weights.h5')
print('Done')
