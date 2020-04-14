# import tensorflow as tf
# from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import concatenate
# from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import Flatten
# from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import ZeroPadding2D

import utils


# Second generation Convolutional ConnectX inspired network
# This version is more flexible on the level of where data flows
def convnet_simple(config):
  input_shape, num_output_channels = utils.get_input_output_shapes(config)
  inputs = [Input(input_shape, dtype='float32', name='input_array')]
  x = inputs[0]
  
  # Convolutional layers
  for conv_output_layer_id, (filters, kernel) in enumerate(
      config['filters_kernels']):
    x = Conv2D(filters=filters, kernel_size=kernel, strides=1,
               padding='same', activation='linear')(x)
    x = Activation('relu')(x)

  # MLP layers
  mlp_layers = config['action_mlp_layers'] + [num_output_channels]
  for i, layer_size in enumerate(mlp_layers):
    if i < (len(mlp_layers)-1):
      # Non final fully connected layers
      x = Dense(layer_size, activation='linear')(x)
      x = Activation('relu')(x)
    else:
      # Sigmoid activation on the final Q-value function output layer
      x = Dense(layer_size, activation='linear')(x)
      outputs = [Activation('sigmoid')(x)]
    
  return inputs, outputs


###############################################
################# UNET MODELS #################
###############################################

def BatchActivate(x, use_layer_norm=True):
  if use_layer_norm:
     x = LayerNormalization()(x)
  else:
    x = BatchNormalization()(x)
  x = Activation('relu')(x)
  return x

def convolution_block(x, filters, size, strides=(1,1), padding='same',
                      activation=True):
  x = Conv2D(filters, size, strides=strides, padding=padding)(x)
  if activation == True:
    x = BatchActivate(x)
  return x

def residual_block(blockInput, num_filters=16, batch_activate=False):
  x = BatchActivate(blockInput)
  x = convolution_block(x, num_filters, (3,3) )
  x = convolution_block(x, num_filters, (3,3), activation=False)
  x = Add()([x, blockInput])
  if batch_activate:
    x = BatchActivate(x)
  return x


def unet_core(inputs, start_neurons, dropout_ratio):
  # 16 -> 8
  conv1 = Conv2D(start_neurons*1, (3, 3), activation=None, padding="same")(
      inputs)
  # conv1 = residual_block(conv1, start_neurons*1)
  conv1 = residual_block(conv1, start_neurons*1, True)
  pool1 = MaxPooling2D((2, 2))(conv1)
  pool1 = Dropout(dropout_ratio/2)(pool1)

  # 8 -> 4
  conv2 = Conv2D(start_neurons*2, (3, 3), activation=None, padding="same")(
      pool1)
  # conv2 = residual_block(conv2, start_neurons*2)
  conv2 = residual_block(conv2, start_neurons*2, True)
  pool2 = MaxPooling2D((2, 2))(conv2)
  pool2 = Dropout(dropout_ratio)(pool2)
  convm_input = pool2
  
  # Middle: 4 -> 4
  convm = Conv2D(start_neurons*4, (3, 3), activation=None, padding="same")(
      convm_input)
  # convm = residual_block(convm, start_neurons*4)
  convm = residual_block(convm, start_neurons*4, True)
  
  # 4 -> 8
  deconv2 = Conv2DTranspose(start_neurons*2, (3, 3), strides=(2, 2),
                            padding="same")(convm)
  uconv2 = concatenate([deconv2, conv2])
      
  uconv2 = Dropout(dropout_ratio)(uconv2)
  uconv2 = Conv2D(start_neurons*2, (3, 3), activation=None, padding="same")(
      uconv2)
  # uconv2 = residual_block(uconv2, start_neurons*2)
  uconv2 = residual_block(uconv2, start_neurons*2, True)
  
  # 8 -> 16
  deconv1 = Conv2DTranspose(start_neurons*1, (3, 3), strides=(2, 2),
                            padding="same")(uconv2)
  uconv1 = concatenate([deconv1, conv1])
  
  uconv1 = Dropout(dropout_ratio)(uconv1)
  uconv1 = Conv2D(start_neurons*1, (3, 3), activation=None, padding="same")(
      uconv1)
  # uconv1 = residual_block(uconv1, start_neurons*1)
  uconv1 = residual_block(uconv1, start_neurons*1, True)
  
  return uconv1

# Initial Unet architecture for Halite
# Pad the inputs with zeros so that the input is 16*16
def padded_unet(config):
  unet_start_neurons = config['unet_start_neurons']
  unet_dropout_ratio = config['unet_dropout_ratio']
  input_shape, num_output_channels = utils.get_input_output_shapes(config)
  inputs = [Input(input_shape, dtype='float32', name='input_array')]
  x = inputs[0]
  
  # Pad the inputs with zeros
  x = ZeroPadding2D()(x)
  x = Cropping2D(((1, 0), (1, 0)))(x)
  
  # Unet
  x = unet_core(x, unet_start_neurons, unet_dropout_ratio)

  # MLP layers
  mlp_layers = config['action_mlp_layers'] + [num_output_channels]
  for i, layer_size in enumerate(mlp_layers):
    if i < (len(mlp_layers)-1):
      # Non final fully connected layers
      x = Dense(layer_size, activation='linear')(x)
      x = Activation('relu')(x)
    else:
      # Sigmoid activation on the final Q-value function output layer
      x = Dense(layer_size, activation='linear')(x)
      outputs = [Activation('sigmoid')(x)]
    
  return inputs, outputs