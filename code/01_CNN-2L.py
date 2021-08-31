import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import metrics
from sklearn.metrics import roc_curve, auc
from pcam_utils import load_norm_data
import logging


# written by Eric Bonnet 
# eric.bonnet@cea.fr 
# very simple CNN model with 2 convolutional layers for the pcam dataset

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level = logging.INFO)

logging.info("loading data")
(x_train, x_valid, x_test, y_train, y_valid, y_test) = load_norm_data(False)

# input image dimensions
img_rows, img_cols = 96,96
input_shape = (img_rows, img_cols, 3)

nb_epochs = 15 
verbose = 1 

batch_size = 32 
nb_dense_layers = 256 
learning_rate = 0.001

### tuner hyperparameters optimized values
#batch_size = 96 
#nb_dense_layers = 128 
#learning_rate = 0.0003476665074399067

print("nb epochs: "+str(nb_epochs))
print("batch size: "+str(batch_size))
print("nb dense layers: "+str(nb_dense_layers))

logging.info("building model")

model = Sequential()

# 1st conv => relu => pool
model.add(Conv2D(32, kernel_size=(5,5), padding="same", input_shape=input_shape))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# 2nd conv => relu => pool
model.add(Conv2D(64, kernel_size=(5,5), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

# flatten => relu layers
model.add(Flatten())
model.add(Dense(nb_dense_layers))
model.add(Activation("relu"))

# final binary layer 
model.add(Dense(1, activation="sigmoid"))

# compile and display model
model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate), metrics=['accuracy'])
model.summary()
print("nb layers: "+str(len(model.layers)))

# use checkpointing to save best weights
checkpoint_path = "weights.h5"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
callbacks_list = [checkpoint]

logging.info("training model")

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs, verbose=verbose, validation_data=(x_valid, y_valid), callbacks=callbacks_list)

logging.info("training done")

# load best weights
model.load_weights(checkpoint_path)

logging.info("evaluate model")
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: '+str(score[0]))
print('Test accuracy: '+str(score[1]))

# predict and calculate ROC
y_pred = model.predict(x_test)
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print("ROC auc: "+str(roc_auc))



