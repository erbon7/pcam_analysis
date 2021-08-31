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

# written by Eric Bonnet
# eric.bonnet@cea.fr 
# fine-tune some hyperparameters (batch size, learning rate) for InceptionV3 model architecture 

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level = logging.INFO)

logging.info("loading data")
(x_train, x_valid, x_test, y_train, y_valid, y_test) = load_norm_data(False)

# input image dimensions
img_rows, img_cols = 96,96
input_shape = (img_rows, img_cols, 3)


def build_model(hp):
   
    # base pre-trained model with imagenet weights
    base_model = InceptionV3(weights='imagenet', include_top=False)

    # global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # fully-connected layer
    x = Dense(256, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    # re-train full model 
    for layer in base_model.layers:
        layer.trainable = True 

    # try different learning rates
    model.compile(loss="binary_crossentropy", optimizer=Adam(hp.Float('learning_rate',min_value=1e-5, max_value=1e-2,sampling='LOG',default=1e-3)), metrics=["accuracy"])

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
    hyperband_iterations=3,
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

