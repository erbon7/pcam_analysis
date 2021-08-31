import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Flatten, Dropout
from tensorflow.keras.optimizers import SGD, Adadelta, Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import metrics
from sklearn.metrics import roc_curve, auc
from pcam_utils import load_norm_data
import logging
from kerastuner.tuners import RandomSearch, Hyperband

# written by Eric Bonnet 
# eric.bonnet@cea.fr 
# fine-tune some hyperparameters (batch size, dense layers size, learning rate) for 
# a CNN with 6 convolutional layers

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level = logging.INFO)

logging.info("loading data")
(x_train, x_valid, x_test, y_train, y_valid, y_test) = load_norm_data(False)

# input image dimensions
img_rows, img_cols = 96,96
input_shape = (img_rows, img_cols, 3)

# CNN-6L architecture
def build_model(hp):
    
    image_size = 96
    kernel_size = (3,3)
    pool_size= (2,2)
    first_filters = 32
    second_filters = 64
    third_filters = 128
    dropout_conv = 0.3
    dropout_dense = 0.5

    model = Sequential()
   
    model.add(Conv2D(first_filters, kernel_size, activation = 'relu', input_shape = (image_size, image_size, 3)))
    model.add(Conv2D(first_filters, kernel_size, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size = pool_size))
    model.add(Dropout(dropout_conv))

    model.add(Conv2D(second_filters, kernel_size, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(second_filters, kernel_size, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size = pool_size))
    model.add(Dropout(dropout_conv))

    model.add(Conv2D(third_filters, kernel_size, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))

    model.add(Conv2D(third_filters, kernel_size, use_bias=False))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size = pool_size))
    model.add(Dropout(dropout_conv))
     
    model.add(Flatten())
    # try different number of nodes for final dense layer
    model.add(Dense(hp.Choice(f'n_nodes', values=[128, 256, 512, 1024]))) 
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dropout(dropout_dense))

    model.add(Dense(1, activation="sigmoid"))

    # try different learning rate values
    model.compile(loss=binary_crossentropy, optimizer=Adam(hp.Float('learning_rate',min_value=1e-5, max_value=1e-2,sampling='LOG',default=1e-3)), metrics=['accuracy'])

    return model

# subclass Hyperband to try different batch sizes
class MyTuner(Hyperband):
  def run_trial(self, trial, *args, **kwargs):
      kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 256, step=32)
      super(MyTuner, self).run_trial(trial, *args, **kwargs)

## Hyperband model
tuner = MyTuner(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    hyperband_iterations=4,
    executions_per_trial=1,  
    directory='__pcam_tuner',
    project_name='pcam_Tuner')

NUM_EPOCH = 15 
tuner.search(x=x_train,
             y=y_train,
             epochs=NUM_EPOCH,
             verbose = 2,
             validation_data=(x_valid, y_valid))

print(tuner.get_best_models()[0].summary())
print(tuner.get_best_hyperparameters()[0].values) 

