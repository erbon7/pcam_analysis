from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from pcam_utils import plot_figures, load_norm_data
from tensorflow.keras import metrics
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import logging

from tensorflow.keras.applications.inception_v3 import InceptionV3

# written by Eric Bonnet 03.2020
# eric.d.bonnet@gmail.com
# transfer learning and full Inception re-training model for the pcam dataset

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level = logging.INFO)

logging.info("loading data")
(x_train, x_valid, x_test, y_train, y_valid, y_test) = load_norm_data()

# Hyper parameters
nb_epochs = 10 
batch_size = 64 
nb_dense_layers = 256
data_augmentation = False 

print("nb epochs: "+str(nb_epochs))
print("batch size: "+str(batch_size))
print("nb dense layers: "+str(nb_dense_layers))
print("data augmentation: "+str(data_augmentation))

if data_augmentation == True:

    train_datagen = ImageDataGenerator(
        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        vertical_flip = True,
        fill_mode = 'nearest')

    train_generator = train_datagen.flow(x_train, y_train, batch_size = batch_size)

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
x = Dense(nb_dense_layers, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
# if this value is set to True, the model will be fully re-trained on the dataset
for layer in base_model.layers:
    layer.trainable = False 

# set checkpointing
checkpoint_path = "pcam_weights.h5"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
callbacks_list = [checkpoint]

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

print(model.summary())
print("nb layers: "+str(len(model.layers)))

logging.info("training model")

if data_augmentation == False:
    history = model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=nb_epochs, batch_size=batch_size, verbose=1, callbacks=callbacks_list)
else:
    steps_per_epoch = x_train.shape[0] // batch_size
    print("steps_per_epoch: "+str(steps_per_epoch))
    history = model.fit_generator(train_generator, 
                                  steps_per_epoch = steps_per_epoch, 
                                  epochs = nb_epochs, 
                                  verbose=1 , 
                                  validation_data = (x_valid, y_valid), 
                                  callbacks = callbacks_list)

logging.info("training done")

# load best weights
model.load_weights(checkpoint_path)

logging.info("evaluate model")

# calculate loss and accuracy on test set
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss: '+str(score[0]))
print('Test accuracy: '+str(score[1]))

# calculate false positive rate, true positive rate, roc area under the curve and plot figures
y_pred = model.predict(x_test)
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print("ROC auc: "+str(roc_auc))

logging.info("plotting figures")

plot_figures(fpr, tpr, history, roc_auc, "roc.png", "loss.png", "accuracy.png")


