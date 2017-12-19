# Keras
import keras as ks
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv1D, Dense
from keras.layers import MaxPooling1D, UpSampling1D, Dropout
from keras.layers import Reshape, LeakyReLU, Lambda

# Tensorboard
from keras.callbacks import TensorBoard
from time import time

# Function definitions

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# Custom loss function

def vae_loss(y_true, y_pred):
    mse_loss = ks.metrics.mean_squared_error(y_true[:,0], y_pred[:,0])
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return mse_loss + kl_loss

# Define params
dropout_rate = 0.1 #OPTIMAL = 0.10
leak = 0.1
latent_dim = 8
epsilon_std = 1.0

# 
x = Input(shape=(512, 1))
x_reshape = Reshape((512,))(x)
### Encoding structure ###

# First encoding layer
h1 = Dense(256, activation = 'linear')(x_reshape)
lr1 = LeakyReLU(leak)(h1)

# Compressed representation
z_mean = Dense(latent_dim)(lr1)
z_log_var = Dense(latent_dim)(lr1)

z = Lambda(sampling)([z_mean, z_log_var])

### Decoding structure ###

# First decoding layer
h2 = Dense(256, activation = 'linear')(z)
lr2 = LeakyReLU(leak)(h2)

out_dense = Dense(512, activation = 'linear')(lr2)
out_lr = LeakyReLU(leak)(out_dense)
out = Reshape((512,1))(out_lr)

model = Model(x,out)

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
        batch_size = 100,
        epochs = 10,
        validation_data = (X_val, Y_val),
        callbacks = [tensorboard]
        )