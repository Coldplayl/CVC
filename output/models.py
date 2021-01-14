from __future__ import division
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
from keras.optimizers import *
from keras.layers import *         
from keras.backend import expand_dims
import keras
from keras.regularizers import l2
from keras.layers import GlobalAveragePooling2D, Reshape, Dense, multiply, Permute, Subtract
from keras import layers
import cv2
import tensorflow as tf
from tensorflow.keras import backend as K
from attention import ChannelWiseAttention1,ChannelWiseAttention2,SpatialAttention1,SpatialAttention2

def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool._keras_shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool._keras_shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat._keras_shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), dilation_rate=1, activation='relu', name=None):

    '''
    2D Convolutional layers
    
    Arguments:
        x {keras layer} -- input layer 
        filters {int} -- number of filters
        num_row {int} -- number of rows in filters
        num_col {int} -- number of columns in filters
    
    Keyword Arguments:
        padding {str} -- mode of padding (default: {'same'})
        strides {tuple} -- stride of convolution operation (default: {(1, 1)})
        activation {str} -- activation function (default: {'relu'})
        name {str} -- name of the layer (default: {None})
    
    Returns:
        [keras layer] -- [output layer]
    '''

    x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    if(activation == None):
        return x

    x = Activation(activation, name=name)(x)

    return x


def RFB(inputs,out_channel):
    branch0 = conv2d_bn(inputs, out_channel,1,1,padding='same')
    
    branch1 = conv2d_bn(inputs, out_channel,1,1,padding='same')
    branch1 = conv2d_bn(branch1, out_channel,1,3,padding='same')
    branch1 = conv2d_bn(branch1,out_channel,3,1,padding='same')
    branch1 = conv2d_bn(branch1,out_channel,3,3,padding='same',dilation_rate=3)
    
    branch2 = conv2d_bn(inputs, out_channel,1,1,padding='same')
    branch2 = conv2d_bn(branch2, out_channel,1,5,padding='same')
    branch2 = conv2d_bn(branch2,out_channel,5,1,padding='same')
    branch2 = conv2d_bn(branch2, out_channel,3,3,padding='same',dilation_rate=5)
    
    branch3 = conv2d_bn(inputs, out_channel,1,1,padding='same')
    branch3 = conv2d_bn(branch3, out_channel,1,7,padding='same')
    branch3 = conv2d_bn(branch3, out_channel,7,1,padding='same')
    branch3 = conv2d_bn(branch3, out_channel,3,3,padding='same',dilation_rate=7)
    
    branch_cat = conv2d_bn(concatenate([branch0,branch1,branch2,branch3],axis=3),out_channel,3,3,padding='same')
    output = Add()([branch_cat,conv2d_bn(inputs,out_channel,1,1,padding='same')])
    output = Activation('relu')(output)
    return output
  
  
def Our_v1(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Dropout(0.1)(pool1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)


    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Dropout(0.1)(pool2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Dropout(0.1)(pool3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Dropout(0.1)(pool4)
    conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)


    up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = BatchNormalization()(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = BatchNormalization()(merge7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = BatchNormalization()(merge8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = BatchNormalization()(merge9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(input=inputs, output=conv10)          
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model   

def Our_v2(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Dropout(0.1)(pool1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)


    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Dropout(0.1)(pool2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Dropout(0.1)(pool3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Dropout(0.1)(pool4)
    # D1
    conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5_1 = Dropout(0.5)(conv5_1)
    # D2
    conv5_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop5_1)
    conv5_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5_2)
    drop5_2 = Dropout(0.5)(conv5_2)
    
    merge_dense = concatenate([conv5_2,drop5_1], axis = 3)
    conv5_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_dense)     
    conv5_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5_3)
    conv5 = Dropout(0.5)(conv5_3)

    up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = BatchNormalization()(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = BatchNormalization()(merge7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = BatchNormalization()(merge8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = BatchNormalization()(merge9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(input=inputs, output=conv10)          
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model 
 
def Our_v3(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    #conv1 = RFB(conv1,16)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Dropout(0.1)(pool1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    #conv2 = RFB(conv2,32)


    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Dropout(0.1)(pool2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = RFB(conv3, 64)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Dropout(0.1)(pool3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = RFB(conv4,128)

    
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Dropout(0.1)(pool4)
    conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = RFB(conv5,128)
    # D1
 #   conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
 #   conv5_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
 #   drop5_1 = Dropout(0.5)(conv5_1)
    # D2
 #   conv5_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(drop5_1)
 #   conv5_2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5_2)
 #   drop5_2 = Dropout(0.5)(conv5_2)
    
 #   merge_dense = concatenate([conv5_2,drop5_1], axis = 3)
 #   conv5_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_dense)     
 #   conv5_3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5_3)
 #   conv5 = Dropout(0.5)(conv5_3)

    up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = BatchNormalization()(merge6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = BatchNormalization()(merge7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = BatchNormalization()(merge8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = BatchNormalization()(merge9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(input=inputs, output=conv10)          
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model 

     

def Our_v4(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Dropout(0.1)(pool1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)


    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Dropout(0.1)(pool2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)


    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Dropout(0.1)(pool3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

 
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Dropout(0.1)(pool4)
    conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    conv5_up = UpSampling2D(size=(2, 2))(conv5)
    conv5_mul = multiply([conv5_up,conv4])
    conv6 = concatenate([conv5_up,conv5_mul],axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    
    conv6_up = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    conv6_mul = multiply([conv6_up,conv3])
    conv7 = concatenate([conv6_up,conv6_mul],axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    
    conv7_up = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv7))
    conv7_mul = multiply([conv7_up,conv2])
    conv8 = concatenate([conv7_up,conv7_mul],axis=3)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)  
    conv8 = BatchNormalization()(conv8)
    
    conv8_up = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv8))
    conv8_mul = multiply([conv8_up,conv1])
    conv9 = concatenate([conv8_up,conv8_mul],axis=3)
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)  
    conv9 = BatchNormalization()(conv9)
    

    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(input=inputs, output=conv10)          
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model    
      
def Our_v5(input_size=(256, 256, 1)):
    inputs = Input(input_size)
    conv1 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Dropout(0.1)(pool1)
    conv2 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)


    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Dropout(0.1)(pool2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = RFB(conv3,64)


    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Dropout(0.1)(pool3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = RFB(conv4,128)

 
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Dropout(0.1)(pool4)
    conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = RFB(conv5,128)

    conv5_up = UpSampling2D(size=(2, 2))(conv5)
    conv5_mul = multiply([conv5_up,conv4])
    conv6 = concatenate([conv5_up,conv5_mul],axis=3)
    conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    
    conv6_up = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    conv6_mul = multiply([conv6_up,conv3])
    conv7 = concatenate([conv6_up,conv6_mul],axis=3)
    conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    
    conv7_up = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv7))
    conv7_mul = multiply([conv7_up,conv2])
    conv8 = concatenate([conv7_up,conv7_mul],axis=3)
    conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)  
    conv8 = BatchNormalization()(conv8)
    
    conv8_up = Conv2D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv8))
    conv8_mul = multiply([conv8_up,conv1])
    conv9 = concatenate([conv8_up,conv8_mul],axis=3)
    conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)  
    conv9 = BatchNormalization()(conv9)
    

    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
    model = Model(input=inputs, output=conv10)          
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model 
