# Keras
import keras as ks
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, MaxPooling1D, UpSampling1D, Dropout, LeakyReLU, Lambda

# Tensorboard
from keras.callbacks import TensorBoard
from time import time

# Function definitions

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# Custom loss layer

def vae_loss(y_true, y_pred):
    mse_loss = ks.metrics.mean_squared_error(y_true, y_pred)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(mse_loss + kl_loss)

# Define dropout rate
dropout_rate = 0.1 #OPTIMAL = 0.10
leak = 0.1

# 
x = Input(shape=(512, 1))
### Encoding structure ###

# First encoding layer
c1 = Conv1D(input_shape = (512, 1),
                 filters = 8,
                 activation = 'linear',
                 kernel_size = 6,
                 strides = 1,
                 padding = 'same'
                )(x)
lr1 = LeakyReLU(leak)(c1)
mp1 = MaxPooling1D(pool_size = 4, padding = 'same')(lr1)
d1 = Dropout(dropout_rate)(mp1)

# Second encoding layer
c2 = Conv1D(filters = 8,
                 activation = 'linear',
                 kernel_size = 6,
                 strides = 1,
                 padding = 'same'
                )(d1)
lr2 = LeakyReLU(leak)(c2)
mp2 = MaxPooling1D(pool_size = 4, padding = 'same')(lr2)
d2 = Dropout(dropout_rate)(mp2)

# Third encoding layer
c3 = Conv1D(filters = 8,
                 activation = 'linear',
                 kernel_size = 6,
                 strides = 1,
                 padding = 'same'
                )(d2)
lr3 = LeakyReLU(leak)(c3)
mp3 = MaxPooling1D(pool_size = 4, padding = 'same')(lr3)
d3 = Dropout(dropout_rate)(mp3)

# Compressed representation
z_mean = Dense(64)(d3)
z_log_var = Dense(64)(d3)

z = Lambda(sampling, output_shape=(64,))([z_mean, z_log_var])

### Decoding structure ###

# First decoding layer
c4 = Conv1D(filters = 8,
                 activation = 'linear',
                 kernel_size = 6,
                 strides = 1,
                 padding = 'same'
                )(d3)
lr4 = LeakyReLU(leak)(c4)
us4 = UpSampling1D(4)(lr4)
d4 = Dropout(dropout_rate)(us4)


# Second decoding layer
c5 = Conv1D(filters = 8,
                 activation = 'linear',
                 kernel_size = 6,
                 strides = 1,
                 padding = 'same'
                )(d4)
lr5 = LeakyReLU(leak)(c5)
us5 = UpSampling1D(4)(lr5)
d5 = Dropout(dropout_rate)(us5)

# Third decoding layer
c6 = Conv1D(filters = 8,
                 activation = 'linear',
                 kernel_size = 6,
                 strides = 1,
                 padding = 'same'
                )(d5)
lr6 = LeakyReLU(leak)(c6)
us6 = UpSampling1D(4)(lr6)
d6 = Dropout(dropout_rate)(us6)

# Final layer
c7 = Conv1D(filters = 1,
                 activation = 'linear',
                 kernel_size = 6,
                 strides = 1,
                 padding = 'same'
                )(d6)
lr7 = LeakyReLU(leak)(c7)
d7 = Dropout(dropout_rate)(lr7)

model = Model(x,d7)

# Compile
model.compile(loss = 'mse',
              optimizer = 'adam',
              metrics = ['mse']
             )

# Tensorboard
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# Training

def train(model):
	model.fit(
		X_train,
        Y_train,
        batch_size = 64,
        epochs = 100,
        validation_data = (X_val, Y_val),
        callbacks = [tensorboard]
        )