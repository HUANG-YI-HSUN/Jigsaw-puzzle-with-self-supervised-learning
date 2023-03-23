# This pgogram can't work, our customed loss function didn't finish
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import tensorflow as tf
import keras
import itertools
# import cv2
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
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops



'''def test(i, j) :
  if ( i > j ) :
    print('fuck') 
    
    
z = tf.convert_to_tensor([0], dtype=tf.int64)
thirtynine = tf.convert_to_tensor([39], dtype=tf.int64)
tf.cond((tf.equal( z,thirtynine )), lambda:test(10, 5), lambda:print('ff'))'''
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


    
def PictureMinus(arrA, arrB):
    arrA = np.array(arrA)
    arrB = np.array(arrB)
    arrA = tf.convert_to_tensor(arrA, dtype=tf.float32)
    arrB = tf.convert_to_tensor(arrB, dtype=tf.float32)
    tf.reshape(arrA, [-1])
    tf.reshape(arrB, [-1])
    square = tf.square( tf.subtract(arrA, arrB ) )
    total = tf.reduce_mean( square )
    print(total,'total')
    return total

    
def ListMinus(pred_index, val_x, left_kind ) :
  left_kind_list = []
  for i in range (0, 40) :
    a = []
    for j in range(0, 9) :
      a.append(left_kind[j][i])
    left_kind_list.append(a)
    
  print(len(left_kind_list))
  lossl = tf.stack([PictureMinus(val_x, left_kind_list[0]), PictureMinus(val_x, left_kind_list[1]), PictureMinus(val_x, left_kind_list[2]), PictureMinus(val_x, left_kind_list[3]), 
                     PictureMinus(val_x, left_kind_list[4]), PictureMinus(val_x, left_kind_list[5]), PictureMinus(val_x, left_kind_list[6]), PictureMinus(val_x, left_kind_list[7]),
                     PictureMinus(val_x, left_kind_list[8]), PictureMinus(val_x, left_kind_list[9]), PictureMinus(val_x, left_kind_list[10]), PictureMinus(val_x, left_kind_list[11]),
                     PictureMinus(val_x, left_kind_list[12]), PictureMinus(val_x, left_kind_list[13]), PictureMinus(val_x, left_kind_list[14]), PictureMinus(val_x, left_kind_list[15]),
                     PictureMinus(val_x, left_kind_list[16]), PictureMinus(val_x, left_kind_list[17]), PictureMinus(val_x, left_kind_list[18]), PictureMinus(val_x, left_kind_list[19]),
                     PictureMinus(val_x, left_kind_list[20]), PictureMinus(val_x, left_kind_list[21]), PictureMinus(val_x, left_kind_list[22]), PictureMinus(val_x, left_kind_list[23]),
                     PictureMinus(val_x, left_kind_list[24]), PictureMinus(val_x, left_kind_list[25]), PictureMinus(val_x, left_kind_list[26]), PictureMinus(val_x, left_kind_list[27]),
                     PictureMinus(val_x, left_kind_list[28]), PictureMinus(val_x, left_kind_list[29]), PictureMinus(val_x, left_kind_list[30]), PictureMinus(val_x, left_kind_list[31]),
                     PictureMinus(val_x, left_kind_list[32]), PictureMinus(val_x, left_kind_list[33]), PictureMinus(val_x, left_kind_list[34]), PictureMinus(val_x, left_kind_list[35]),
                     PictureMinus(val_x, left_kind_list[36]), PictureMinus(val_x, left_kind_list[37]), PictureMinus(val_x, left_kind_list[38]), PictureMinus(val_x, left_kind_list[39])], axis=0)
    
  print(lossl)    
  return lossl

def Assign(val_x, left_kind, i ):    
  for j in range(0,9):
    val_x[j] = left_kind[j][i]

def GiveValue(left_kind, temparray, numlist, index ) :
  for i in range(0,len(numlist)) :
    j = numlist[i]
    left_kind[i][index] = temparray[j]

def Loss(y_true, y_pred): 
            i = 0
            val_x = [] 
            val_x.append(np.full((1, 64, 64, 3), 0.0000001))
            val_x.append(np.full((1, 64, 64, 3), 0.0000001))
            val_x.append(np.full((1, 64, 64, 3), 0.0000001))
            val_x.append(np.full((1, 64, 64, 3), 0.0000001))
            val_x.append(np.full((1, 64, 64, 3), 0.0000001))
            val_x.append(np.full((1, 64, 64, 3), 0.0000001))
            val_x.append(np.full((1, 64, 64, 3), 0.0000001))
            val_x.append(np.full((1, 64, 64, 3), 0.0000001))
            val_x.append(np.full((1, 64, 64, 3), 0.0000001))
            
            left_kind = []
            left_kind.append(np.full((40, 64, 64, 3), 0.0000001))
            left_kind.append(np.full((40, 64, 64, 3), 0.0000001))
            left_kind.append(np.full((40, 64, 64, 3), 0.0000001))
            left_kind.append(np.full((40, 64, 64, 3), 0.0000001))
            left_kind.append(np.full((40, 64, 64, 3), 0.0000001))
            left_kind.append(np.full((40, 64, 64, 3), 0.0000001))
            left_kind.append(np.full((40, 64, 64, 3), 0.0000001))
            left_kind.append(np.full((40, 64, 64, 3), 0.0000001))
            left_kind.append(np.full((40, 64, 64, 3), 0.0000001))
            
            temp_array = picture_list
            random = tf.argmax(y_pred, 1)
            true =tf.argmax(y_true, 1)
            print(random, 'random')
            print(true, 'true')
            
            numlist = [0,1,2,3,4,5,6,7,8]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([0], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
         
            numlist = [8,7,6,5,4,3,2,1,0]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([1], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
            
            numlist = [0,1,2,6,7,8,3,4,5]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([2], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
              
            numlist = [3,4,5,0,1,2,6,7,8]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([3], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
              
            numlist = [3,4,5,6,7,8,0,1,2]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([4], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
            
            numlist = [6,7,8,0,1,2,3,4,5]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([5], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
            
            numlist = [6,7,8,3,4,5,0,1,2]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([6], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
            
            numlist = [0,2,1,3,5,4,6,8,7]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([7], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
            
            numlist = [1,0,2,4,3,5,7,6,8]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([8], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
            
            numlist = [2,1,0,5,4,3,8,7,6]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([9], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
            
            numlist = [2,0,1,5,3,4,8,6,7]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([10], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
            
            numlist = [1,2,0,4,5,3,7,8,6]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([11], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
        
            numlist = [7,0,1,5,8,4,3,6,2]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([12], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
            
            numlist = [5,0,1,7,2,4,3,6,8]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([13], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
            
            numlist = [7,0,1,5,6,4,3,8,2]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([14], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
            
            numlist = [0,2,1,7,8,4,3,6,5]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([15], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
            
            numlist = [5,0,1,4,2,7,3,6,8]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([16], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
            
            numlist = [5,0,1,3,4,8,7,6,2]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([17], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
            
            numlist = [5,0,1,8,7,3,4,2,6]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([18], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
            
            numlist = [6,0,1,7,8,4,3,5,2]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([19], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
            
            numlist = [5,6,7,8,4,3,2,1,0]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([20], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
            
            numlist = [0,1,2,3,4,8,7,6,5]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([21], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
        
            numlist = [5,1,2,3,4,8,7,6,0]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([22], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
        
            numlist = [0,1,2,3,8,4,7,6,5]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([23], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
        
            numlist = [1,0,3,2,8,4,7,6,5]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([24], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
        
            numlist = [0,1,2,3,8,4,7,5,6]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([25], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
        
            numlist = [3,1,2,0,8,4,7,6,5]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([26], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
        
            numlist = [0,1,2,7,8,4,3,6,5]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([27], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
        
            numlist = [8,7,6,2,1,0,5,4,3]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([28], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1

            numlist = [5,4,3,8,7,6,2,1,0]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([29], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1

            numlist = [5,4,3,2,1,0,8,7,6]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([30], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1

            numlist = [2,1,0,8,7,6,5,4,3]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([31], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1

            numlist = [2,1,0,5,4,3,8,7,6]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([32], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1

            numlist = [7,6,8,3,4,5,0,1,2]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([33], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1

            numlist = [8,6,7,5,3,4,2,0,1]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([34], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1

            numlist = [7,8,6,4,5,3,1,2,0]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([35], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1

            numlist = [7,6,8,4,3,5,1,0,2]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([36], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1

            numlist = [6,8,7,3,5,4,0,2,1]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([37], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1

            numlist = [8,1,6,3,4,5,2,7,0]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([38], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1

            numlist = [0,7,2,5,4,3,6,1,8]
            GiveValue(left_kind, temp_array, numlist, i )
            z = tf.convert_to_tensor([39], dtype=tf.int64)
            tf.cond((tf.equal( z,random )), lambda:Assign(val_x, left_kind, i), lambda:print('no'))
            i = i + 1
            
            return ListMinus(random, val_x, left_kind)    
            
def RetrunLoss():
  lossl = np.array([])
  for i in range(0, 40) :
    lossl=np.append(lossl,0.0000001)
  lossl = tf.convert_to_tensor(lossl, dtype=tf.float32)
  print(lossl, 'lossl')
  return lossl
   
def custom_loss(picture_list): #Custom loss function
    def loss(y_true, y_pred):
        yp = tf.argmax(y_pred, 1)
        yt = tf.argmax(y_true, 1)
        tf.cond((tf.equal( yt,yp )), lambda:RetrunLoss(), lambda:Loss(y_true, y_pred))
        
    return loss
    
    
def ff() :
  def custom_loss_function(y_true, y_pred):
     squared_difference = tf.square(y_true - y_pred)
     s= tf.reduce_mean(squared_difference, axis=-1) 
     print(s, 'this is s')
     return s
  a =  custom_loss_function
        
  return a

# Get data
num_crops = 9
num_permutations = 100
picture_list = []


P_hat = np.array(list(itertools.permutations(list(range(num_crops)), num_crops)))
new_array = np.zeros( (100,9) )

for i in range(num_permutations) :
    n = P_hat.shape[0]
    index = np.random.randint(n)
    np.delete(P_hat, index, axis=0)
    new_array[i] = P_hat[index]

train_test_split = 0.7

#Read all the folders in the directory

#Declare training array
cat_list = []
x = []
y = []
y_label = 0

#Using just 5 images per category

rootdir = "/data/data9_50000"
file_names = [[]for i in range(9)]
filenames = ''


for i in range (1 , 10) : 
    for parent, dirnames, filenames in os.walk("/data/data9_50000/data9/" + str(i)):  
        file_names[i-1] = filenames
        file_names[i-1] = sorted(file_names[i-1])
    print(len(file_names[i-1]))

train_size = int(len(file_names[0]) * 0.7)
print(len(file_names[0]), 'size')



def get_batch (data_index, batch_size, random_data, index):
        global picture_list
        train_length = 1 * batch_size
        start=0
        end=train_size
        batch_x=[]
        batch_y = np.zeros((1, 40))
    
        i = 0
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
        random_data_index = 0
   
        for j in range ( 1 , 10 ) :
            a = keras.preprocessing.image.load_img("/data/data9_50000/data9/"+str(j)+"/"+file_names[j-1][random_data[data_index] * 1 + random_data_index], 
                                                    grayscale=False, color_mode="rgb", target_size=(64, 64), interpolation="nearest" )
    
            temp_array[j-1] = keras.preprocessing.image.img_to_array(a)
            temp_array[j-1] = np.asarray(temp_array[j-1])/255.0
            picture_list.append(temp_array[j-1])
        random_data_index = random_data_index + 1
        
        if ( index == 0 ) :
          batch_y[0][0] = 1
          batch_x[0][i] = temp_array[0]
          batch_x[1][i] = temp_array[1]
          batch_x[2][i] = temp_array[2]
          batch_x[3][i] = temp_array[3]
          batch_x[4][i] = temp_array[4]
          batch_x[5][i] = temp_array[5]
          batch_x[6][i] = temp_array[6] 
          batch_x[7][i] = temp_array[7]
          batch_x[8][i] = temp_array[8]
          return(batch_x, batch_y)    
        
        if ( index == 1 ) : 
          batch_y[0][1] = 1
          batch_x[0][i] = temp_array[8]
          batch_x[1][i] = temp_array[7]
          batch_x[2][i] = temp_array[6]
          batch_x[3][i] = temp_array[5]
          batch_x[4][i] = temp_array[4]
          batch_x[5][i] = temp_array[3]
          batch_x[6][i] = temp_array[2] 
          batch_x[7][i] = temp_array[1]
          batch_x[8][i] = temp_array[0]
          return(batch_x, batch_y)    
        
        if ( index==2) :
          batch_y[0][2] = 1
          batch_x[0][i] = temp_array[0]
          batch_x[1][i] = temp_array[1]
          batch_x[2][i] = temp_array[2]
          batch_x[3][i] = temp_array[6]
          batch_x[4][i] = temp_array[7]
          batch_x[5][i] = temp_array[8]
          batch_x[6][i] = temp_array[3] 
          batch_x[7][i] = temp_array[4]
          batch_x[8][i] = temp_array[5]
          return(batch_x, batch_y)    
          
        if ( index == 3 ) :
          batch_y[0][3] = 1
          batch_x[0][i] = temp_array[3]
          batch_x[1][i] = temp_array[4]
          batch_x[2][i] = temp_array[5]
          batch_x[3][i] = temp_array[0]
          batch_x[4][i] = temp_array[1]
          batch_x[5][i] = temp_array[2]
          batch_x[6][i] = temp_array[6] 
          batch_x[7][i] = temp_array[7]
          batch_x[8][i] = temp_array[8]
          return(batch_x, batch_y)    
        
        if ( index == 4 ) :
          batch_y[0][4] = 1
          batch_x[0][i] = temp_array[3]
          batch_x[1][i] = temp_array[4]
          batch_x[2][i] = temp_array[5]
          batch_x[3][i] = temp_array[6]
          batch_x[4][i] = temp_array[7]
          batch_x[5][i] = temp_array[8]
          batch_x[6][i] = temp_array[0] 
          batch_x[7][i] = temp_array[1]
          batch_x[8][i] = temp_array[2]
          return(batch_x, batch_y)    
        
        if ( index == 5 ) :
          batch_y[0][5] = 1
          batch_x[0][i] = temp_array[6]
          batch_x[1][i] = temp_array[7]
          batch_x[2][i] = temp_array[8]
          batch_x[3][i] = temp_array[0]
          batch_x[4][i] = temp_array[1]
          batch_x[5][i] = temp_array[2]
          batch_x[6][i] = temp_array[3] 
          batch_x[7][i] = temp_array[4]
          batch_x[8][i] = temp_array[5]
          return(batch_x, batch_y)    
        
        if ( index == 6 ) :
          batch_y[0][6] = 1
          batch_x[0][i] = temp_array[6]
          batch_x[1][i] = temp_array[7]
          batch_x[2][i] = temp_array[8]
          batch_x[3][i] = temp_array[3]
          batch_x[4][i] = temp_array[4]
          batch_x[5][i] = temp_array[5]
          batch_x[6][i] = temp_array[0] 
          batch_x[7][i] = temp_array[1]
          batch_x[8][i] = temp_array[2]
          return(batch_x, batch_y)    
        
        if ( index == 7 ) :
          batch_y[0][7] = 1
          batch_x[0][i] = temp_array[0]
          batch_x[1][i] = temp_array[2]
          batch_x[2][i] = temp_array[1]
          batch_x[3][i] = temp_array[3]
          batch_x[4][i] = temp_array[5]
          batch_x[5][i] = temp_array[4]
          batch_x[6][i] = temp_array[6] 
          batch_x[7][i] = temp_array[8]
          batch_x[8][i] = temp_array[7]
          return(batch_x, batch_y)    
        
        if ( index == 8 ):
          batch_y[0][8] = 1
          batch_x[0][i] = temp_array[1]
          batch_x[1][i] = temp_array[0]
          batch_x[2][i] = temp_array[2]
          batch_x[3][i] = temp_array[4]
          batch_x[4][i] = temp_array[3]
          batch_x[5][i] = temp_array[5]
          batch_x[6][i] = temp_array[7] 
          batch_x[7][i] = temp_array[6]
          batch_x[8][i] = temp_array[8]
          return(batch_x, batch_y)    
        
        if ( index == 9 ) :
          batch_y[0][9] = 1
          batch_x[0][i] = temp_array[2]
          batch_x[1][i] = temp_array[1]
          batch_x[2][i] = temp_array[0]
          batch_x[3][i] = temp_array[5]
          batch_x[4][i] = temp_array[4]
          batch_x[5][i] = temp_array[3]
          batch_x[6][i] = temp_array[8] 
          batch_x[7][i] = temp_array[7]
          batch_x[8][i] = temp_array[6]
          return(batch_x, batch_y)    
        
        if ( index == 10 ):
          batch_y[0][10] = 1
          batch_x[0][i] = temp_array[2]
          batch_x[1][i] = temp_array[0]
          batch_x[2][i] = temp_array[1]
          batch_x[3][i] = temp_array[5]
          batch_x[4][i] = temp_array[3]
          batch_x[5][i] = temp_array[4]
          batch_x[6][i] = temp_array[8] 
          batch_x[7][i] = temp_array[6]
          batch_x[8][i] = temp_array[7]
          return(batch_x, batch_y)    
        
        if ( index == 11 ):
          batch_y[0][11] = 1
          batch_x[0][i] = temp_array[1]
          batch_x[1][i] = temp_array[2]
          batch_x[2][i] = temp_array[0]
          batch_x[3][i] = temp_array[4]
          batch_x[4][i] = temp_array[5]
          batch_x[5][i] = temp_array[3]
          batch_x[6][i] = temp_array[7] 
          batch_x[7][i] = temp_array[8]
          batch_x[8][i] = temp_array[6]
          return(batch_x, batch_y)    
        
        if ( index == 12 ):
          batch_y[0][12] = 1
          batch_x[0][i] = temp_array[7]
          batch_x[1][i] = temp_array[0]
          batch_x[2][i] = temp_array[1]
          batch_x[3][i] = temp_array[5]
          batch_x[4][i] = temp_array[8]
          batch_x[5][i] = temp_array[4]
          batch_x[6][i] = temp_array[3] 
          batch_x[7][i] = temp_array[6]
          batch_x[8][i] = temp_array[2]
          return(batch_x, batch_y)    
        
        if ( index == 13 ):
          batch_y[0][13] = 1
          batch_x[0][i] = temp_array[5]
          batch_x[1][i] = temp_array[0]
          batch_x[2][i] = temp_array[1]
          batch_x[3][i] = temp_array[7]
          batch_x[4][i] = temp_array[2]
          batch_x[5][i] = temp_array[4]
          batch_x[6][i] = temp_array[3] 
          batch_x[7][i] = temp_array[6]
          batch_x[8][i] = temp_array[8]
          return(batch_x, batch_y)    
        
        if ( index == 14 ):
          batch_y[0][14] = 1
          batch_x[0][i] = temp_array[7]
          batch_x[1][i] = temp_array[0]
          batch_x[2][i] = temp_array[1]
          batch_x[3][i] = temp_array[5]
          batch_x[4][i] = temp_array[6]
          batch_x[5][i] = temp_array[4]
          batch_x[6][i] = temp_array[3] 
          batch_x[7][i] = temp_array[8]
          batch_x[8][i] = temp_array[2]
          return(batch_x, batch_y)    
        
        if ( index == 15):
          batch_y[0][15] = 1
          batch_x[0][i] = temp_array[0]
          batch_x[1][i] = temp_array[2]
          batch_x[2][i] = temp_array[1]
          batch_x[3][i] = temp_array[7]
          batch_x[4][i] = temp_array[8]
          batch_x[5][i] = temp_array[4]
          batch_x[6][i] = temp_array[3] 
          batch_x[7][i] = temp_array[6]
          batch_x[8][i] = temp_array[5]
          return(batch_x, batch_y)    
        
        if ( index == 16 ):
          batch_y[0][16] = 1
          batch_x[0][i] = temp_array[5]
          batch_x[1][i] = temp_array[0]
          batch_x[2][i] = temp_array[1]
          batch_x[3][i] = temp_array[4]
          batch_x[4][i] = temp_array[2]
          batch_x[5][i] = temp_array[7]
          batch_x[6][i] = temp_array[3] 
          batch_x[7][i] = temp_array[6]
          batch_x[8][i] = temp_array[8]
          return(batch_x, batch_y)    
        
        if ( index == 17 ):
          batch_y[0][17] = 1
          batch_x[0][i] = temp_array[5]
          batch_x[1][i] = temp_array[0]
          batch_x[2][i] = temp_array[1]
          batch_x[3][i] = temp_array[3]
          batch_x[4][i] = temp_array[4]
          batch_x[5][i] = temp_array[8]
          batch_x[6][i] = temp_array[7] 
          batch_x[7][i] = temp_array[6]
          batch_x[8][i] = temp_array[2]
          return(batch_x, batch_y)    
        
        if ( index == 18 ):
          batch_y[0][18] = 1
          batch_x[0][i] = temp_array[5]
          batch_x[1][i] = temp_array[0]
          batch_x[2][i] = temp_array[1]
          batch_x[3][i] = temp_array[8]
          batch_x[4][i] = temp_array[7]
          batch_x[5][i] = temp_array[3]
          batch_x[6][i] = temp_array[4] 
          batch_x[7][i] = temp_array[2]
          batch_x[8][i] = temp_array[6]
          return(batch_x, batch_y)    
        
        if ( index == 19 ):
          batch_y[0][19] = 1
          batch_x[0][i] = temp_array[6]
          batch_x[1][i] = temp_array[0]
          batch_x[2][i] = temp_array[1]
          batch_x[3][i] = temp_array[7]
          batch_x[4][i] = temp_array[8]
          batch_x[5][i] = temp_array[4]
          batch_x[6][i] = temp_array[3] 
          batch_x[7][i] = temp_array[5]
          batch_x[8][i] = temp_array[2]
          return(batch_x, batch_y)    
        
        if ( index == 20 ):
          batch_y[0][20] = 1
          batch_x[0][i] = temp_array[5]
          batch_x[1][i] = temp_array[6]
          batch_x[2][i] = temp_array[7]
          batch_x[3][i] = temp_array[8]
          batch_x[4][i] = temp_array[4]
          batch_x[5][i] = temp_array[3]
          batch_x[6][i] = temp_array[2] 
          batch_x[7][i] = temp_array[1]
          batch_x[8][i] = temp_array[0]
          return(batch_x, batch_y)    
          
        if ( index == 21 ):
          batch_y[0][21] = 1
          batch_x[0][i] = temp_array[0]
          batch_x[1][i] = temp_array[1]
          batch_x[2][i] = temp_array[2]
          batch_x[3][i] = temp_array[3]
          batch_x[4][i] = temp_array[4]
          batch_x[5][i] = temp_array[8]
          batch_x[6][i] = temp_array[7] 
          batch_x[7][i] = temp_array[6]
          batch_x[8][i] = temp_array[5]
          return(batch_x, batch_y)    
        
        if ( index == 22 ):
          batch_y[0][22] = 1
          batch_x[0][i] = temp_array[5]
          batch_x[1][i] = temp_array[1]
          batch_x[2][i] = temp_array[2]
          batch_x[3][i] = temp_array[3]
          batch_x[4][i] = temp_array[4]
          batch_x[5][i] = temp_array[8]
          batch_x[6][i] = temp_array[7] 
          batch_x[7][i] = temp_array[6]
          batch_x[8][i] = temp_array[0]
          return(batch_x, batch_y)    
        
        if ( index == 23 ):
          batch_y[0][23] = 1
          batch_x[0][i] = temp_array[0]
          batch_x[1][i] = temp_array[1]
          batch_x[2][i] = temp_array[2]
          batch_x[3][i] = temp_array[3]
          batch_x[4][i] = temp_array[8]
          batch_x[5][i] = temp_array[4]
          batch_x[6][i] = temp_array[7] 
          batch_x[7][i] = temp_array[6]
          batch_x[8][i] = temp_array[5]
          return(batch_x, batch_y)    
        
        if ( index == 24 ):
          batch_y[0][24] = 1
          batch_x[0][i] = temp_array[1]
          batch_x[1][i] = temp_array[0]
          batch_x[2][i] = temp_array[3]
          batch_x[3][i] = temp_array[2]
          batch_x[4][i] = temp_array[8]
          batch_x[5][i] = temp_array[4]
          batch_x[6][i] = temp_array[7] 
          batch_x[7][i] = temp_array[6]
          batch_x[8][i] = temp_array[5]
          return(batch_x, batch_y)    
        
        if ( index == 25 ):
          batch_y[0][25] = 1
          batch_x[0][i] = temp_array[0]
          batch_x[1][i] = temp_array[1]
          batch_x[2][i] = temp_array[2]
          batch_x[3][i] = temp_array[3]
          batch_x[4][i] = temp_array[8]
          batch_x[5][i] = temp_array[4]
          batch_x[6][i] = temp_array[7] 
          batch_x[7][i] = temp_array[5]
          batch_x[8][i] = temp_array[6]
          return(batch_x, batch_y)    
        
        if ( index == 26 ):
          batch_y[0][26] = 1
          batch_x[0][i] = temp_array[3]
          batch_x[1][i] = temp_array[1]
          batch_x[2][i] = temp_array[2]
          batch_x[3][i] = temp_array[0]
          batch_x[4][i] = temp_array[8]
          batch_x[5][i] = temp_array[4]
          batch_x[6][i] = temp_array[7] 
          batch_x[7][i] = temp_array[6]
          batch_x[8][i] = temp_array[5]
          return(batch_x, batch_y) 
        
        if ( index == 27 ):
          batch_y[0][27] = 1
          batch_x[0][i] = temp_array[0]
          batch_x[1][i] = temp_array[1]
          batch_x[2][i] = temp_array[2]
          batch_x[3][i] = temp_array[7]
          batch_x[4][i] = temp_array[8]
          batch_x[5][i] = temp_array[4]
          batch_x[6][i] = temp_array[3] 
          batch_x[7][i] = temp_array[6]
          batch_x[8][i] = temp_array[5]
          return(batch_x, batch_y)    
        
        if (index == 28) :
          batch_y[0][28] = 1
          batch_x[0][i] = temp_array[8]
          batch_x[1][i] = temp_array[7]
          batch_x[2][i] = temp_array[6]
          batch_x[3][i] = temp_array[2]
          batch_x[4][i] = temp_array[1]
          batch_x[5][i] = temp_array[0]
          batch_x[6][i] = temp_array[5] 
          batch_x[7][i] = temp_array[4]
          batch_x[8][i] = temp_array[3]
          return(batch_x, batch_y)    

        if ( index == 29 ):
          batch_y[0][29] = 1
          batch_x[0][i] = temp_array[5]
          batch_x[1][i] = temp_array[4]
          batch_x[2][i] = temp_array[3]
          batch_x[3][i] = temp_array[8]
          batch_x[4][i] = temp_array[7]
          batch_x[5][i] = temp_array[6]
          batch_x[6][i] = temp_array[2] 
          batch_x[7][i] = temp_array[1]
          batch_x[8][i] = temp_array[0]
          return(batch_x, batch_y)    

        if ( index == 30 ):
          batch_y[0][30] = 1
          batch_x[0][i] = temp_array[5]
          batch_x[1][i] = temp_array[4]
          batch_x[2][i] = temp_array[3]
          batch_x[3][i] = temp_array[2]
          batch_x[4][i] = temp_array[1]
          batch_x[5][i] = temp_array[0]
          batch_x[6][i] = temp_array[8] 
          batch_x[7][i] = temp_array[7]
          batch_x[8][i] = temp_array[6]
          return(batch_x, batch_y)    

        if ( index == 31 ):
          batch_y[0][31] = 1
          batch_x[0][i] = temp_array[2]
          batch_x[1][i] = temp_array[1]
          batch_x[2][i] = temp_array[0]
          batch_x[3][i] = temp_array[8]
          batch_x[4][i] = temp_array[7]
          batch_x[5][i] = temp_array[6]
          batch_x[6][i] = temp_array[5] 
          batch_x[7][i] = temp_array[4]
          batch_x[8][i] = temp_array[3]
          return(batch_x, batch_y)    

        if ( index == 32 ):
          batch_y[0][32] = 1
          batch_x[0][i] = temp_array[2]
          batch_x[1][i] = temp_array[1]
          batch_x[2][i] = temp_array[0]
          batch_x[3][i] = temp_array[5]
          batch_x[4][i] = temp_array[4]
          batch_x[5][i] = temp_array[3]
          batch_x[6][i] = temp_array[8] 
          batch_x[7][i] = temp_array[7]
          batch_x[8][i] = temp_array[6]
          return(batch_x, batch_y)    

        if ( index == 33 ):
          batch_y[0][33] = 1
          batch_x[0][i] = temp_array[7]
          batch_x[1][i] = temp_array[6]
          batch_x[2][i] = temp_array[8]
          batch_x[3][i] = temp_array[3]
          batch_x[4][i] = temp_array[4]
          batch_x[5][i] = temp_array[5]
          batch_x[6][i] = temp_array[0] 
          batch_x[7][i] = temp_array[1]
          batch_x[8][i] = temp_array[2]
          return(batch_x, batch_y)    

        if ( index == 34 ) : 
          batch_y[0][34] = 1
          batch_x[0][i] = temp_array[8]
          batch_x[1][i] = temp_array[6]
          batch_x[2][i] = temp_array[7]
          batch_x[3][i] = temp_array[5]
          batch_x[4][i] = temp_array[3]
          batch_x[5][i] = temp_array[4]
          batch_x[6][i] = temp_array[2] 
          batch_x[7][i] = temp_array[0]
          batch_x[8][i] = temp_array[1]
          return(batch_x, batch_y)    

        if ( index == 35 ) :
          batch_y[0][35] = 1
          batch_x[0][i] = temp_array[7]
          batch_x[1][i] = temp_array[8]
          batch_x[2][i] = temp_array[6]
          batch_x[3][i] = temp_array[4]
          batch_x[4][i] = temp_array[5]
          batch_x[5][i] = temp_array[3]
          batch_x[6][i] = temp_array[1] 
          batch_x[7][i] = temp_array[2]
          batch_x[8][i] = temp_array[0]
          return(batch_x, batch_y)    

        if ( index == 36 ) :
          batch_y[0][36] = 1
          batch_x[0][i] = temp_array[7]
          batch_x[1][i] = temp_array[6]
          batch_x[2][i] = temp_array[8]
          batch_x[3][i] = temp_array[4]
          batch_x[4][i] = temp_array[3]
          batch_x[5][i] = temp_array[5]
          batch_x[6][i] = temp_array[1] 
          batch_x[7][i] = temp_array[0]
          batch_x[8][i] = temp_array[2]
          return(batch_x, batch_y)    

        if ( index == 37 ):
          batch_y[0][37] = 1
          batch_x[0][i] = temp_array[6]
          batch_x[1][i] = temp_array[8]
          batch_x[2][i] = temp_array[7]
          batch_x[3][i] = temp_array[3]
          batch_x[4][i] = temp_array[5]
          batch_x[5][i] = temp_array[4]
          batch_x[6][i] = temp_array[0] 
          batch_x[7][i] = temp_array[2]
          batch_x[8][i] = temp_array[1]
          return(batch_x, batch_y)    

        if ( index == 38 ):
          batch_y[0][38] = 1
          batch_x[0][i] = temp_array[8]
          batch_x[1][i] = temp_array[1]
          batch_x[2][i] = temp_array[6]
          batch_x[3][i] = temp_array[3]
          batch_x[4][i] = temp_array[4]
          batch_x[5][i] = temp_array[5]
          batch_x[6][i] = temp_array[2] 
          batch_x[7][i] = temp_array[7]
          batch_x[8][i] = temp_array[0]
          return(batch_x, batch_y)    

        if ( index == 39 ):
          batch_y[0][39] = 1
          batch_x[0][i] = temp_array[0]
          batch_x[1][i] = temp_array[7]
          batch_x[2][i] = temp_array[2]
          batch_x[3][i] = temp_array[5]
          batch_x[4][i] = temp_array[4]
          batch_x[5][i] = temp_array[3]
          batch_x[6][i] = temp_array[6] 
          batch_x[7][i] = temp_array[1]
          batch_x[8][i] = temp_array[8]
          return(batch_x, batch_y)   
           
          del temp_array

#Building a model
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

a = 1
b = 0
siamese_net.compile(loss=custom_loss(picture_list), optimizer=optimizer, metrics='mse')

# siamese_net.load_weights('my_model_weights.h5')
siamese_net.summary()
# plot_model(siamese_net, show_shapes=True, show_layer_names=True)


def get_val(val_data_index, batch_size, random_data, val_ii): 
    val_length = batch_size
    val_x = []
    val_y = np.zeros((val_length, 40))
    #np.random.shuffle(batch_y)

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
            a = keras.preprocessing.image.load_img("/data/data9_50000/data9/"+str(j)+"/"+file_names[j-1][random_data[val_data_index] * 1 + val_ii], 
                                                    grayscale=False, color_mode="rgb", target_size=(64, 64), interpolation="nearest" )
    
            temp_array[j-1] = keras.preprocessing.image.img_to_array(a)
            temp_array[j-1] = np.asarray(temp_array[j-1])/255.0
             
        
        val_data_index = val_data_index + 1
        random = np.random.randint(40)
                
        if(random == 0) :
            val_y[i][0] = 1
        # choice = np.random.randint(100)
        # batch_y[i][choice] = 1
        # print(new_array[choice][0])
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
        # choice = np.random.randint(100)
        # batch_y[i][choice] = 1
        # print(new_array[choice][0])
            val_x[0][i] = temp_array[8]
            val_x[1][i] = temp_array[7]
            val_x[2][i] = temp_array[6]
            val_x[3][i] = temp_array[5]
            val_x[4][i] = temp_array[4]
            val_x[5][i] = temp_array[3]
            val_x[6][i] = temp_array[2] 
            val_x[7][i] = temp_array[1]
            val_x[8][i] = temp_array[0]
        #If train_y has 0 pick from the same class, else pick from any other class
        
        elif(random == 2) :
            val_y[i][2] = 1
        # choice = np.random.randint(100)
        # batch_y[i][choice] = 1
        # print(new_array[choice][0])
            val_x[0][i] = temp_array[0]
            val_x[1][i] = temp_array[1]
            val_x[2][i] = temp_array[2]
            val_x[3][i] = temp_array[6]
            val_x[4][i] = temp_array[7]
            val_x[5][i] = temp_array[8]
            val_x[6][i] = temp_array[3] 
            val_x[7][i] = temp_array[4]
            val_x[8][i] = temp_array[5]
        #If train_y has 0 pick from the same class, else pick from any other class

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

    del temp_array
    return(val_x, val_y)


epochs = 21
n_way = 20
n_val = 100
valsize = 500
# print( 'class list', np.random.randint(train_size+1, len(folder_list)-1, n_val))
batch_size = 1

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
i = 0


for epoch in range(1,epochs):
    data_index = 0
    val_data_index = int(len(file_names[0])/batch_size * 0.7)
    random_data = random.sample( range(0, int(len(file_names[0])/batch_size)), int(len(file_names[0])/batch_size))
    for each_epoch in range (0, 1 ) : 
        batch_x, batch_y = get_batch(data_index, batch_size, random_data, i)
        loss, acc = siamese_net.train_on_batch(batch_x, batch_y) 
        train_loss_total = train_loss_total + loss
        train_acc_total = train_acc_total + acc
        i = i + 1
        if ( i == 40 ) :
          i = 0
          data_index = data_index + 1
        print('Batch:', each_epoch, ',  Loss:',loss, ',      Accuracy:', acc)
        del batch_x
        del batch_y
        if ( each_epoch % 100 == 0 ) :
            gc.collect()
        picture_list.clear()
    train_loss_total = train_loss_total / each_round
    train_acc_total = train_acc_total / each_round
    train_loss_his.append( str(train_loss_total) )
    loss_his.append( " " )
    train_acc_his.append( str(train_acc_total) )
    accuracy_his.append( " " )  
  
    print('-----------------------------')
    for k in range(0, val_each_round) : 
        i = 0
        val_x, val_y = get_val(val_data_index,batch_size, random_data, val_ii)
        val_ii = val_ii + 1
        if ( i == 40 ) :
          i = 0
          data_index = data_index + 1
        if ( val_ii == 10 ) :
          val_ii = 0
          val_data_index = val_data_index + 1
        val_loss, val_acc = siamese_net.test_on_batch( val_x, val_y )
        val_loss_total = val_loss_total + val_loss
        acc_loss_total = acc_loss_total + val_acc
        del val_x
        del val_y
        if ( each_epoch % 100 == 0 ) :
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
    gc.collect()


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