import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.optimizers import SGD, Adadelta, Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import metrics
from sklearn.metrics import roc_curve, auc
from pcam_utils import save_data, load_norm_data
import logging
from kerastuner.tuners import RandomSearch

# written by Eric Bonnet 03.2020
# eric.d.bonnet@gmail.com
# very simple CNN model with 2 convolutional layers for the pcam dataset

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level = logging.INFO)

logging.info("loading data")
(x_train, x_valid, x_test, y_train, y_valid, y_test) = load_norm_data(False)

#x_train = x_train[:120000,]
#y_train = y_train[:120000,]

# input image dimensions
img_rows, img_cols = 96,96
input_shape = (img_rows, img_cols, 3)

#nb_epochs = 15 
#batch_size = 32 
#nb_dense_layers = 256 
#verbose = 1 

#logging.info("building model")

def build_model(hp):
    
    model = Sequential()
    
    # 1st conv => relu => pool
    model.add(Conv2D(32, kernel_size=(5,5), padding="same", input_shape=input_shape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 2nd conv => relu => pool
    model.add(Conv2D(64, kernel_size=(5,5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(hp.Choice(f'n_nodes', values=[128, 256, 512, 1024]))) 
    model.add(Activation('relu')) 

    model.add(Dense(1, activation="sigmoid"))

    model.compile(loss=binary_crossentropy, optimizer=Adam(hp.Float('learning_rate',min_value=1e-5, max_value=1e-2,sampling='LOG',default=1e-3)), metrics=['accuracy'])

    return model


class MyTuner(RandomSearch):
  def run_trial(self, trial, *args, **kwargs):
      # You can add additional HyperParameters for preprocessing and custom training loops
      # via overriding `run_trial`
      kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 256, step=32)
      super(MyTuner, self).run_trial(trial, *args, **kwargs)




#tuner = RandomSearch(
tuner = MyTuner(
    build_model,
    objective='val_accuracy',
    max_trials=30,  # how many model variations to test?
    executions_per_trial=1,  # how many trials per variation? (same model could perform differently)
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

