# Keras
import keras as ks
from keras.models import Sequential
from keras.layers import Conv1D, Dense, MaxPooling1D, UpSampling1D, Dropout, LeakyReLU

# Tensorboard
from keras.callbacks import TensorBoard
from time import time


# Define dropout rate
dropout_rate = 0.1 #OPTIMAL = 0.15
leak = 0.1

# 
model = Sequential()

### Encoding structure ###

# First encoding layer
model.add(Conv1D(input_shape = (512, 1),
                 filters = 8,
                 activation = 'linear',
                 kernel_size = 6,
                 strides = 1,
                 padding = 'same'
                ))
model.add(LeakyReLU(leak))
model.add(MaxPooling1D(pool_size = 4, padding = 'same'))
model.add(Dropout(dropout_rate))

# Second encoding layer
model.add(Conv1D(filters = 8,
                 activation = 'linear',
                 kernel_size = 6,
                 strides = 1,
                 padding = 'same'
                ))
model.add(LeakyReLU(leak))
model.add(MaxPooling1D(pool_size = 4, padding = 'same'))
model.add(Dropout(dropout_rate))

# Third encoding layer
model.add(Conv1D(filters = 8,
                 activation = 'linear',
                 kernel_size = 6,
                 strides = 1,
                 padding = 'same'
                ))
model.add(LeakyReLU(leak))
model.add(MaxPooling1D(pool_size = 4, padding = 'same'))
model.add(Dropout(dropout_rate))

# Compressed representation
#model.add(Dense(256))
#model.add(LeakyReLU(leak))
#model.add(Dropout(dropout_rate))

### Decoding structure ###

# First decoding layer
model.add(Conv1D(filters = 8,
                 activation = 'linear',
                 kernel_size = 6,
                 strides = 1,
                 padding = 'same'
                ))
model.add(LeakyReLU(leak))
model.add(UpSampling1D(4))
model.add(Dropout(dropout_rate))

# Second decoding layer
model.add(Conv1D(filters = 8,
                 activation = 'linear',
                 kernel_size = 6,
                 strides = 1,
                 padding = 'same'
                ))
model.add(LeakyReLU(leak))
model.add(UpSampling1D(4))
model.add(Dropout(dropout_rate))

# Third decoding layer
model.add(Conv1D(filters = 8,
                 activation = 'linear',
                 kernel_size = 6,
                 strides = 1,
                 padding = 'same'
                ))
model.add(LeakyReLU(leak))
model.add(UpSampling1D(4))
model.add(Dropout(dropout_rate))

# Final layer
model.add(Conv1D(filters = 1,
                 activation = 'linear',
                 kernel_size = 6,
                 strides = 1,
                 padding = 'same'
                ))
model.add(LeakyReLU(leak))
model.add(Dropout(dropout_rate))

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