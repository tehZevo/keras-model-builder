from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import optimizers
import numpy as np
import fire

def dense_stack(input_shape, sizes=[64, 64], activation="swish"):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    for size in sizes:
        model.add(Dense(size, activation=activation))
    return model

def convpool2_stack(input_shape, sizes=[32, 32], kernel_shape=[3, 3], pool_size=[2, 2], activation="swish"):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    for size in sizes:
        model.add(Conv2D(size, kernel_shape, padding="same", activation=activation))
        model.add(MaxPooling2D(pool_size, padding='same'))
    return model

def convpool2up_stack(input_shape, sizes=[32, 32], kernel_shape=[3, 3], pool_size=[2, 2], activation="swish"):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    for size in sizes:
        model.add(Conv2D(size, kernel_shape, padding="same", activation=activation))
        model.add(UpSampling2D(pool_size))
    return model

def dense_ae(input_shape, dense_sizes=[64, 64], latent_size=8, activation="swish", latent_activation="tanh"):
    inputs = Input(input_shape)
    x = inputs
    x = dense_stack(x.shape[1:], dense_sizes, activation)(x)
    x = Dense(latent_size, activation=latent_activation)(x)
    encoder = Model(inputs, x)

    inputs = Input([latent_size])
    x = inputs
    x = dense_stack(x.shape[1:], reversed(dense_sizes), activation)(x)
    x = Dense(input_shape[-1], activation=activation)(x)
    decoder = Model(inputs, x)

    inputs = Input(input_shape)
    x = inputs
    x = decoder(encoder(x))

    autoencoder = Model(inputs, x)
    return autoencoder

def conv2_ae(input_shape, conv_sizes=[32, 32], dense_sizes=[64], latent_size=8, kernel_shape=[3, 3], pool_size=[2, 2], output_activation="sigmoid", activation="swish", latent_activation="tanh"):
    inputs = Input(input_shape)
    x = inputs
    x = convpool2_stack(x.shape[1:], conv_sizes, kernel_shape, pool_size, activation)(x)
    pre_flatten_shape = x.shape[1:]
    x = Flatten()(x)
    x = dense_stack(x.shape[1:], dense_sizes, activation)(x)
    x = Dense(latent_size, activation=latent_activation)(x)
    encoder = Model(inputs, x)

    inputs = Input([latent_size])
    x = inputs
    x = dense_stack(x.shape[1:], reversed(dense_sizes), activation)(x)
    x = Dense(np.prod(pre_flatten_shape), activation=activation)(x)
    x = Reshape(pre_flatten_shape)(x)
    x = convpool2up_stack(x.shape[1:], reversed(conv_sizes), kernel_shape, pool_size, activation)(x)
    #final conv
    #x = Conv2D(input_shape[-1], kernel_shape, padding="same", activation=output_activation)(x)
    decoder = Model(inputs, x)

    inputs = Input(input_shape)
    x = inputs
    x = decoder(encoder(x))

    autoencoder = Model(inputs, x)
    return autoencoder

class Builder:
  def __init__(self):
    self.model = None

  #TODO: DRY this out
  def dense_stack(self, *args, **kwargs):
    self.model = dense_stack(*args, **kwargs)
    return self

  def convpool2_stack(self, *args, **kwargs):
    self.model = convpool2_stack(*args, **kwargs)
    return self

  def convpool2up_stack(self, *args, **kwargs):
    self.model = convpool2up_stack(*args, **kwargs)
    return self

  def dense_ae(self, *args, **kwargs):
    print(args, kwargs)
    self.model = dense_ae(*args, **kwargs)
    return self

  def conv2_ae(self, *args, **kwargs):
    self.model = conv2_ae(*args, **kwargs)
    return self

  def compile(self, loss="mse", optimizer="adam", **kwargs):
    #grab class of named optimizer and pass kwargs to it (eg lr)
    opt = type(optimizers.get(optimizer))(**kwargs)
    self.model.compile(loss=loss, optimizer=opt)
    return self

  def save(self, path="models/model.h5"):
    self.model.save(path)
    return self

  def summary(self):
    print(self.model.summary())
    return self

if __name__ == '__main__':
  fire.Fire(Builder)
