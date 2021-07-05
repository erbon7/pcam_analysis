import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD, Adadelta, Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import metrics
from sklearn.metrics import roc_curve, auc
from pcam_utils import save_data, load_norm_data
import logging
from tensorflow.keras.applications.inception_v3 import InceptionV3
from kerastuner.tuners import RandomSearch, Hyperband

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level = logging.INFO)

logging.info("loading data")
(x_train, x_valid, x_test, y_train, y_valid, y_test) = load_norm_data(False)

#x_train = x_train[:40000,]
#y_train = y_train[:40000,]

# input image dimensions
img_rows, img_cols = 96,96
input_shape = (img_rows, img_cols, 3)

#nb_epochs = 15 
#batch_size = 32 
#nb_dense_layers = 256 
#verbose = 1 

#logging.info("building model")

def build_model(hp):
   
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # let's add a fully-connected layer
    #x = Dense(hp.Choice(f'n_nodes', values=[128, 256, 512, 1024]), activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    # if this value is set to True, the model will be fully re-trained on the dataset
    for layer in base_model.layers:
        layer.trainable = True 

    model.compile(loss="binary_crossentropy", optimizer=Adam(hp.Float('learning_rate',min_value=1e-5, max_value=1e-2,sampling='LOG',default=1e-3)), metrics=["accuracy"])

    return model


#class MyTuner(RandomSearch):
class MyTuner(Hyperband):
  def run_trial(self, trial, *args, **kwargs):
      # You can add additional HyperParameters for preprocessing and custom training loops
      # via overriding `run_trial`
      kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 32, 256, step=32)
      super(MyTuner, self).run_trial(trial, *args, **kwargs)

## randomSearch model
#tuner = MyTuner(
#    build_model,
#    objective='val_accuracy',
#    max_trials=50,  # how many model variations to test?
#    executions_per_trial=1,  # how many trials per variation? (same model could perform differently)
#    directory='__pcam_tuner',
#    project_name='pcam_Tuner')

## Hyperband model
tuner = MyTuner(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    hyperband_iterations=3,
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

