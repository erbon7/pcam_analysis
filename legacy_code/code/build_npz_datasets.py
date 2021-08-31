from keras.utils import HDF5Matrix
import sys
import numpy as np
import os 

# written by Eric Bonnet 10.2019
# eric.d.bonnet@gmail.com
# extract train, validation and test data from pcam HDF5 files and save them in numpy npz file 

# save data as images, labels and npz file 
def process():
    x_data = HDF5Matrix(x_file_name, 'x')
    y_data = HDF5Matrix(y_file_name, 'y')

    Nx = x_data.shape[0]
    Ny = y_data.shape[0]

    print("x data size: " + str(Nx))
    print("y data size: " + str(Ny))

    tmp = []
    y_data = np.array(y_data)
    for i in range(0, len(y_data)):
        tmp.append(int(y_data[i]))
    print("done.")
    y_data = np.array(tmp)
    
    print("saving data as npz file...")

    if image_path == "test":
        np.savez(npz_output, x_test=x_data, y_test=y_data)

    if image_path == "train":
        np.savez(npz_output, x_train=x_data, y_train=y_data)

    if image_path == "valid":
        np.savez(npz_output, x_valid=x_data, y_valid=y_data)

# test data
x_file_name = "camelyonpatch_level_2_split_test_x.h5"
y_file_name = "camelyonpatch_level_2_split_test_y.h5"
npz_output = "pcam_test_data.npz"
image_path = "test"

process()

# train data
x_file_name = "camelyonpatch_level_2_split_train_x.h5"
y_file_name = "camelyonpatch_level_2_split_train_y.h5"
npz_output = "pcam_train_data.npz"
image_path = "train"

process()

# validation data
x_file_name = "camelyonpatch_level_2_split_valid_x.h5"
y_file_name = "camelyonpatch_level_2_split_valid_y.h5"
npz_output = "pcam_valid_data.npz"
image_path = "valid"

process()






