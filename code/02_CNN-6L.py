from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging
from pcam_utils import load_norm_data
from tensorflow.keras import metrics
from sklearn.metrics import roc_curve, auc

# written by Eric Bonnet 
# eric.bonnet@cea.fr
# deep CNN model for the pcam dataset, with 6 convolutional layers

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level = logging.INFO)

logging.info("loading data")
(x_train, x_valid, x_test, y_train, y_valid, y_test) = load_norm_data(False)

# model

batch_size = 32 
nb_dense_layers = 256
learning_rate = 0.001

nb_epochs = 15 
image_size = 96
kernel_size = (3,3)
pool_size= (2,2)
first_filters = 32
second_filters = 64
third_filters = 128
data_augmentation = False 

### tuner hyperparameters optimization values

#batch_size = 64 
#learning_rate = 0.003745751582487425 
#nb_dense_layers = 256 


print("nb epochs: "+str(nb_epochs))
print("batch size: "+str(batch_size))
print("nb dense layers: "+str(nb_dense_layers))
print("data augmentation: "+str(data_augmentation))

dropout_conv = 0.3
dropout_dense = 0.5

if data_augmentation == True:

    train_datagen = ImageDataGenerator(
        rotation_range = 180,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        vertical_flip = True,
        fill_mode = 'nearest')

    train_generator = train_datagen.flow(x_train, y_train, batch_size = batch_size)


logging.info("building model")

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
model.add(Dense(nb_dense_layers, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(dropout_dense))
model.add(Dense(1, activation = "sigmoid"))

# Compile the model
model.compile(Adam(learning_rate), loss = "binary_crossentropy", metrics=["accuracy"])
model.summary()
print("nb layers: "+str(len(model.layers)))

# checkpointing set to save the best model 
checkpoint_path="weights.h5"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_weights_only=True, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

logging.info("training model")

if data_augmentation == False:
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epochs, verbose=1, validation_data=(x_valid, y_valid), callbacks=callbacks_list)
else:
    steps_per_epoch = x_train.shape[0] // batch_size
    print("steps_per_epoch: "+str(steps_per_epoch))
    history = model.fit_generator(train_generator, steps_per_epoch = steps_per_epoch, epochs = nb_epochs, verbose=1 , validation_data = (x_valid, y_valid), callbacks = callbacks_list)

logging.info("training done")

# load best weights
model.load_weights(checkpoint_path)

# evaluation on the test set 
logging.info("evaluate model")

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: '+str(score[0]))
print('Test accuracy: '+str(score[1]))

y_pred = model.predict(x_test)
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print("ROC auc: "+str(roc_auc))


